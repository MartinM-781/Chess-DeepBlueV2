//! Estimation du niveau Elo : duels contre une échelle d'ancres internes de
//! force croissante, puis ajustement par maximum de vraisemblance sur le modèle
//! logistique Elo : p(battre une ancre R_a) = 1 / (1 + 10^((R_a - R) / 400)).
//!
//! Utiliser TOUS les scores (et pas un simple « gagné → ancre suivante ») rend
//! l'estimation bien plus précise à nombre de parties égal : chaque ancre est
//! un point de mesure, la logistique fait l'interpolation.
//!
//! IMPORTANT : les Elo des ancres sont des ESTIMATIONS (échelle maison). La
//! courbe vaut d'abord pour sa TENDANCE ; une calibration contre un moteur UCI
//! à force limitée (Stockfish UCI_Elo) pourra la recaler plus tard.

use std::io::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shakmaty::{Chess, Move};

use crate::arena;
use crate::bots::{Bot, MaterialBot, NetBot, RandomBot};
use crate::nn::Mlp;
use crate::uci::{StockfishBot, UciEngine};

/// Nature de l'adversaire d'une ancre.
pub enum GenreAncre {
    /// Bot maison : None = aléatoire, Some(d) = MaterialBot profondeur d.
    Maison { profondeur: Option<u32> },
    /// Moteur UCI bridé (UCI_LimitStrength + UCI_Elo, clampé aux bornes
    /// annoncées par le moteur), `movetime_ms` ms de réflexion par coup.
    /// Joué SEULEMENT si un chemin moteur est disponible (voir mesure_uci).
    Uci { elo_nominal: u32, movetime_ms: u64 },
}

/// Une ancre de l'échelle : nom, Elo estimé, genre d'adversaire.
pub struct Ancre {
    pub nom: &'static str,
    pub elo: f64,
    pub genre: GenreAncre,
}

/// Échelle mixte, triée par force croissante : le bas (400-1550) reste sur les
/// bots maison (continuité historique), le haut (1700-2300) sur Stockfish
/// bridé — sans ces ancres hautes, le fit sature dès que le réseau écrase
/// tous les bots maison (~1600+) et la courbe perd toute résolution.
///
/// L'ORDRE de cette liste est un point de contrat : la graine d'un duel se
/// dérive du rang de l'ancre PARMI CELLES DE SON GENRE (voir `rang_dans_genre`),
/// jamais de son rang dans la liste jouée. Ajouter une ancre EN FIN de bloc de
/// genre laisse donc les duels historiques bit à bit identiques ; en insérer
/// une au milieu les décalerait tous.
pub const ANCRES: &[Ancre] = &[
    Ancre { nom: "aleatoire", elo: 400.0, genre: GenreAncre::Maison { profondeur: None } },
    Ancre { nom: "materiel d1", elo: 800.0, genre: GenreAncre::Maison { profondeur: Some(1) } },
    Ancre { nom: "materiel d2", elo: 1100.0, genre: GenreAncre::Maison { profondeur: Some(2) } },
    Ancre { nom: "materiel d3", elo: 1350.0, genre: GenreAncre::Maison { profondeur: Some(3) } },
    Ancre { nom: "materiel d4", elo: 1550.0, genre: GenreAncre::Maison { profondeur: Some(4) } },
    Ancre { nom: "stockfish 1700", elo: 1700.0, genre: GenreAncre::Uci { elo_nominal: 1700, movetime_ms: 60 } },
    Ancre { nom: "stockfish 2000", elo: 2000.0, genre: GenreAncre::Uci { elo_nominal: 2000, movetime_ms: 60 } },
    // Plafond relevé : le réseau marquait déjà ~19 % contre l'ancre 2000, une
    // échelle qui s'arrête là ne peut plus mesurer que « au-dessus de tout ».
    Ancre { nom: "stockfish 2300", elo: 2300.0, genre: GenreAncre::Uci { elo_nominal: 2300, movetime_ms: 60 } },
];

/// Score mesuré contre une ancre.
#[derive(Clone, Copy, Debug)]
pub struct MesureAncre {
    pub nom: &'static str,
    pub elo_ancre: f64,
    /// Pourcentage de points dans [0, 1] (victoire 1, nulle 0.5).
    pub score: f64,
    pub parties: usize,
}

// --- Échelle ADAPTATIVE ------------------------------------------------------
//
// Une mesure a un budget de parties fixe. Le dépenser à parts égales sur toute
// l'échelle revient à rejouer des duels dont on connaît déjà l'issue : à
// h225+, l'information de Fisher par ancre (variance de l'estimateur d'Elo
// rapportée à une partie) valait 0,934 pour materiel d4 et 0,916 pour
// stockfish 1700, contre 0,269 / 0,190 / 0,041 / 0,008 pour materiel d3, d2,
// d1 et l'aléatoire — 57 % des parties allaient à quatre ancres saturées.
// Concentrer le même budget sur les seules ancres INFORMATIVES multiplie
// l'information récoltée sans coûter une partie de plus.

/// Borne basse de la plage informative : seuil d'ENTRÉE d'une ancre inactive.
pub const SCORE_INFORMATIF_MIN: f64 = 0.15;
/// Borne haute de la plage informative : seuil d'ENTRÉE d'une ancre inactive.
pub const SCORE_INFORMATIF_MAX: f64 = 0.85;
/// HYSTÉRÉSIS — bande de MAINTIEN, plus large que la bande d'entrée : une ancre
/// DÉJÀ active n'est écartée que si son score sort de [10 %, 90 %].
///
/// Sans elle, la décision se prenait sur un seul tirage bruité : à 24 parties,
/// l'écart-type d'un score est de ~10 points, et le journal réel montre
/// materiel d4 à 72,9 / 0,0 / 43,8 / 85,4 % sur quatre mesures consécutives.
/// L'ancre la PLUS informative de l'échelle (information de Fisher 0,934, devant
/// stockfish 1700 à 0,916) se trouvait ainsi écartée pour 0,4 point au-dessus du
/// couperet — une demi-partie sur 24 — et ne pouvait revenir qu'au re-sondage,
/// des dizaines de mesures plus tard. La bande de maintien absorbe ce bruit :
/// l'entrée reste exigeante (il faut être franchement informatif pour prendre du
/// budget), la sortie devient franche (il faut être franchement saturé pour le
/// perdre). Sur les 200 dernières mesures du journal, cela supprime l'essentiel
/// des bascules d'ensemble d'ancres — bascules qui, l'estimand du MLE dépendant
/// du plan (voir `dispersion_ancres`), déforment la courbe publiée.
pub const SCORE_MAINTIEN_MIN: f64 = 0.10;
/// Borne haute de la bande de maintien (voir `SCORE_MAINTIEN_MIN`).
pub const SCORE_MAINTIEN_MAX: f64 = 0.90;
/// Plancher d'ancres jouées : sous trois points de mesure, le MLE n'a plus de
/// quoi trianguler, et une saturation passagère amputerait toute l'échelle.
pub const ANCRES_ACTIVES_MIN: usize = 3;
/// Une ancre saturée est re-sondée à faible volume toutes les N mesures : le
/// réseau progresse, une ancre sortie de la plage peut y revenir (typiquement
/// une ancre HAUTE qui devient enfin battable).
///
/// L'unité est la MESURE, pas le cycle d'entraînement : la période gouverne un
/// plan de mesure, dont l'horloge naturelle est la mesure (avec --elo-every 8,
/// 25 mesures valent 200 cycles). Le verdict d'un re-sondage n'est PAS gratuit :
/// à faible volume il est presque aussi bruité que ce qu'il tranche, d'où le
/// couple retenu — période courte, volume doublé (voir `PARTIES_RESONDAGE`) —
/// qui garde le coût annuel de l'ancien réglage (50 mesures × 6 parties) en
/// divisant par deux l'écart-type du verdict.
pub const PERIODE_RESONDAGE: u64 = 25;
/// Volume d'un re-sondage (pair : arena alterne les couleurs). À 6 parties,
/// l'écart-type d'un score valait ~20 points : la ré-intégration d'une ancre
/// tenait du tirage au sort (59 % de chances de rentrer pour une ancre vraiment
/// informative à 86 %, 27 % de FAUSSE ré-intégration pour une ancre vraiment
/// saturée à 95 %). 12 parties ramènent cet écart-type à ~14 points.
pub const PARTIES_RESONDAGE: usize = 12;

/// Échéance de re-sondage de l'ancre `index`, en mesures.
///
/// DÉSYNCHRONISÉE d'une ancre à l'autre : les ancres saturées le deviennent
/// souvent à la MÊME mesure (le réseau franchit un palier), leurs compteurs
/// avancent donc en phase et arrivaient à échéance ensemble — une rafale qui
/// prenait d'un coup 5 × 12 parties sur le budget de la mesure et injectait cinq
/// scores extrêmes dans le même journal. Le décalage de 2 mesures par rang
/// étale la dépense sans changer sa fréquence.
pub fn echeance_resondage(index: usize) -> u64 {
    PERIODE_RESONDAGE + 2 * index as u64
}

/// Une ligne du plan de mesure : quelle ancre, combien de parties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanAncre {
    /// Index dans la liste d'ancres (`ANCRES` en production).
    pub index: usize,
    /// Parties allouées à cette ancre pour CETTE mesure.
    pub parties: usize,
    /// Vrai si l'ancre n'est jouée qu'au titre du re-sondage périodique d'une
    /// ancre saturée (faible volume) — utile aux journaux, pas au calcul.
    pub resondage: bool,
}

/// Rang d'une ancre parmi celles de MÊME genre, dans l'ordre de `ancres`.
/// C'est LUI qui dérive la graine des duels (et non le rang dans le plan) :
/// une ancre écartée par l'échelle adaptative ne décale donc jamais les duels
/// des autres, et une mesure partielle reste comparable à une mesure complète.
/// `pub` : train.rs mesure les mêmes ancres avec son propre agent et doit
/// dériver EXACTEMENT les mêmes graines.
pub fn rang_dans_genre(ancres: &[Ancre], index: usize) -> usize {
    let est_maison = |a: &Ancre| matches!(a.genre, GenreAncre::Maison { .. });
    let meme = est_maison(&ancres[index]);
    ancres[..index].iter().filter(|a| est_maison(a) == meme).count()
}

/// Distance d'un dernier score à la plage informative (0 = dedans, ou jamais
/// mesurée donc prioritaire). Sert à compléter le plancher d'ancres actives
/// par les MOINS saturées.
fn distance_a_la_plage(dernier: Option<f64>) -> f64 {
    match dernier {
        None => 0.0,
        Some(s) if s < SCORE_INFORMATIF_MIN => SCORE_INFORMATIF_MIN - s,
        Some(s) if s > SCORE_INFORMATIF_MAX => s - SCORE_INFORMATIF_MAX,
        Some(_) => 0.0,
    }
}

/// Plan « historique » : toutes les ancres, `parties_par_ancre` parties
/// chacune. C'est ce que faisaient `mesure` et `mesure_uci` avant l'échelle
/// adaptative — elles le construisent toujours, à l'identique.
pub fn plan_complet(ancres: &[Ancre], parties_par_ancre: usize) -> Vec<PlanAncre> {
    (0..ancres.len())
        .map(|index| PlanAncre { index, parties: parties_par_ancre, resondage: false })
        .collect()
}

/// Sélection ADAPTATIVE + répartition du budget.
///
/// - `derniers[i]` : dernier score connu de l'ancre i (None = jamais mesurée) ;
/// - `depuis[i]` : mesures écoulées depuis son dernier duel ;
/// - `total_parties` : budget TOTAL de la mesure (et non par ancre).
///
/// Règles (dans l'ordre) :
/// 1. aucun historique du tout (premier démarrage) → TOUTES les ancres, comme
///    avant ce chantier ;
/// 2. sinon, ancres ACTIVES = dernier score dans la bande qui leur applique —
///    [15 %, 85 %] pour une ancre inactive qui demande à entrer, [10 %, 90 %]
///    pour une ancre déjà active qui demande à rester (HYSTÉRÉSIS, voir
///    `SCORE_MAINTIEN_MIN`) — plus toute ancre encore jamais mesurée (une ancre
///    neuve doit être sondée) ;
/// 3. plancher : si moins de `ANCRES_ACTIVES_MIN`, on complète par les ancres
///    les moins éloignées de la plage ;
/// 4. re-sondage : les saturées inactives depuis `echeance_resondage(i)` mesures
///    reçoivent `PARTIES_RESONDAGE` parties, PRÉLEVÉES sur le total (le budget
///    d'une mesure reste exactement celui demandé) et jamais au point de faire
///    tomber une active sous 2 parties ;
/// 5. le reste se répartit également entre les actives, par PAIRES de parties
///    (arena alterne les couleurs : un compte pair donne autant de blancs que
///    de noirs à l'agent mesuré) ; le surplus va aux premières ancres.
///
/// Les entrées à zéro partie sont omises : une ancre du plan est toujours une
/// ancre réellement jouée.
///
/// `actives_avant` vide = aucune ancre n'était active (bande d'entrée pour
/// toutes) : c'est le cas d'un état vierge et des appels de diagnostic.
pub fn plan_adaptatif(
    ancres: &[Ancre],
    derniers: &[Option<f64>],
    depuis: &[u64],
    total_parties: usize,
) -> Vec<PlanAncre> {
    plan_adaptatif_eligibles(
        ancres,
        derniers,
        depuis,
        &[],
        &vec![true; ancres.len()],
        total_parties,
    )
}

/// `plan_adaptatif` restreint aux ancres ÉLIGIBLES (`eligibles[i]`).
///
/// Une ancre inéligible est une ancre qu'on ne peut PAS jouer dans le contexte
/// courant : les ancres UCI sans moteur (--oracle absent) ou en régime 1-pli,
/// qui ne les a jamais jouées. Les écarter AVANT la sélection — et non après —
/// est ce qui garantit qu'une mesure porte toujours sur des duels réels : sans
/// ce filtre, l'échelle adaptative pouvait concentrer tout le budget sur trois
/// ancres Stockfish (c'est exactement l'état du journal à h238) et rendre une
/// mesure VIDE si le moteur manquait. Le plancher de trois ancres et le
/// re-sondage s'appliquent alors parmi les seules éligibles.
pub fn plan_adaptatif_eligibles(
    ancres: &[Ancre],
    derniers: &[Option<f64>],
    depuis: &[u64],
    actives_avant: &[bool],
    eligibles: &[bool],
    total_parties: usize,
) -> Vec<PlanAncre> {
    let n = ancres.len();
    if n == 0 || total_parties == 0 {
        return Vec::new();
    }
    let dernier = |i: usize| derniers.get(i).copied().flatten();
    let ecoule = |i: usize| depuis.get(i).copied().unwrap_or(0);
    let eligible = |i: usize| eligibles.get(i).copied().unwrap_or(true);
    // Une ancre absente de `actives_avant` est réputée INACTIVE : elle doit
    // franchir la bande d'entrée, la plus exigeante des deux.
    let active_avant = |i: usize| actives_avant.get(i).copied().unwrap_or(false);
    let candidates: Vec<usize> = (0..n).filter(|&i| eligible(i)).collect();
    if candidates.is_empty() {
        return Vec::new();
    }

    // 1-2. Ancres actives, avec HYSTÉRÉSIS : bande de maintien pour celles qui
    //      l'étaient déjà, bande d'entrée pour les autres.
    let aucun_historique = candidates.iter().all(|&i| dernier(i).is_none());
    let mut actives: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&i| {
            aucun_historique
                || match dernier(i) {
                    None => true,
                    Some(s) if active_avant(i) => {
                        (SCORE_MAINTIEN_MIN..=SCORE_MAINTIEN_MAX).contains(&s)
                    }
                    Some(s) => (SCORE_INFORMATIF_MIN..=SCORE_INFORMATIF_MAX).contains(&s),
                }
        })
        .collect();
    // 3. Plancher : compléter par les MOINS saturées (à égalité, la plus basse
    //    dans l'échelle — ordre déterministe).
    let plancher = ANCRES_ACTIVES_MIN.min(candidates.len());
    if actives.len() < plancher {
        let mut restants: Vec<usize> =
            candidates.iter().copied().filter(|i| !actives.contains(i)).collect();
        restants.sort_by(|&a, &b| {
            distance_a_la_plage(dernier(a))
                .total_cmp(&distance_a_la_plage(dernier(b)))
                .then(a.cmp(&b))
        });
        for i in restants {
            if actives.len() >= plancher {
                break;
            }
            actives.push(i);
        }
        actives.sort_unstable();
    }

    // 4. Re-sondage des saturées, prélevé sur le budget.
    let mut resondees: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|i| !actives.contains(i) && ecoule(*i) >= echeance_resondage(*i))
        .collect();
    let plafond_resondage =
        total_parties.saturating_sub(2 * actives.len()) / PARTIES_RESONDAGE;
    resondees.truncate(plafond_resondage);
    let budget_actives = total_parties - resondees.len() * PARTIES_RESONDAGE;

    // 5. Répartition par paires entre les actives.
    let mut plan: Vec<PlanAncre> = Vec::with_capacity(actives.len() + resondees.len());
    if !actives.is_empty() {
        let paires = budget_actives / 2;
        let base = paires / actives.len();
        let surplus = paires % actives.len();
        for (rang, &index) in actives.iter().enumerate() {
            let parties = 2 * (base + usize::from(rang < surplus));
            if parties > 0 {
                plan.push(PlanAncre { index, parties, resondage: false });
            }
        }
    }
    for index in resondees {
        plan.push(PlanAncre { index, parties: PARTIES_RESONDAGE, resondage: true });
    }
    plan.sort_by_key(|e| e.index);
    plan
}

/// Mémoire de l'échelle adaptative : dernier score connu de chaque ancre et
/// ancienneté de ce score. Reconstruite au démarrage depuis `ancres.csv` (le
/// journal que l'entraîneur alimente déjà), puis tenue à jour en mémoire à
/// chaque mesure — un redémarrage ne repart donc PAS à zéro, et aucun nouveau
/// fichier d'état n'apparaît.
#[derive(Clone, Debug)]
pub struct EtatEchelle {
    derniers: Vec<Option<f64>>,
    depuis: Vec<u64>,
    /// L'ancre était-elle ACTIVE à la dernière mesure (jouée à plein volume, et
    /// non au seul titre du re-sondage) ? C'est ce drapeau qui décide laquelle
    /// des deux bandes de l'hystérésis s'applique à la mesure suivante.
    actives: Vec<bool>,
}

impl EtatEchelle {
    /// État vierge (aucun historique) : la première mesure jouera TOUTES les
    /// ancres, comme avant ce chantier.
    pub fn neuf(nb_ancres: usize) -> Self {
        EtatEchelle {
            derniers: vec![None; nb_ancres],
            depuis: vec![0; nb_ancres],
            actives: vec![false; nb_ancres],
        }
    }

    /// Reconstruit l'état depuis un journal `ancres.csv`
    /// (`heures,ancre,score_pct,parties[,resondage]`). Fichier absent, illisible
    /// ou tronqué → état vierge : l'échelle repart complète, jamais d'erreur.
    ///
    /// Toutes les lignes d'une même mesure partagent la valeur `heures` : le
    /// nombre de valeurs DISTINCTES vues après la dernière apparition d'une
    /// ancre donne directement son ancienneté en mesures.
    ///
    /// La 5e colonne (`resondage`, 0/1) est facultative : les lignes historiques
    /// n'en ont pas et valent 0 — c'était alors des mesures à plein volume.
    /// Une ancre jouée à la DERNIÈRE mesure autrement qu'en re-sondage est
    /// rétablie comme ACTIVE : sans cela, un redémarrage ferait perdre à
    /// l'hystérésis toute sa mémoire et rejouerait la bascule qu'elle évite.
    pub fn charge_csv(chemin: &str, ancres: &[Ancre]) -> Self {
        let mut etat = EtatEchelle::neuf(ancres.len());
        let Ok(contenu) = std::fs::read_to_string(chemin) else {
            return etat;
        };
        // Numéro de la mesure (0, 1, ...) où chaque ancre a été vue en dernier.
        let mut vue_a: Vec<Option<u64>> = vec![None; ancres.len()];
        // Cette dernière apparition était-elle un re-sondage ?
        let mut vue_en_resondage: Vec<bool> = vec![false; ancres.len()];
        let mut mesures: u64 = 0;
        let mut heures_courantes: Option<String> = None;
        for ligne in contenu.lines().skip(1) {
            let champs: Vec<&str> = ligne.trim_end().split(',').collect();
            if champs.len() < 3 {
                continue;
            }
            let (heures, nom, score_pct) = (champs[0], champs[1], champs[2]);
            if heures_courantes.as_deref() != Some(heures) {
                if heures_courantes.is_some() {
                    mesures += 1;
                }
                heures_courantes = Some(heures.to_string());
            }
            let Some(i) = ancres.iter().position(|a| a.nom == nom) else {
                continue; // ancre disparue de l'échelle : ignorée
            };
            // Un score non fini (journal corrompu) n'est pas « saturé » : c'est
            // une ligne à ignorer, sinon il fausserait le tri du plancher.
            let Some(pct) = score_pct.parse::<f64>().ok().filter(|p| p.is_finite()) else {
                continue;
            };
            etat.derniers[i] = Some(pct / 100.0);
            vue_a[i] = Some(mesures);
            vue_en_resondage[i] = champs.get(4).map(|c| c.trim() == "1").unwrap_or(false);
        }
        if heures_courantes.is_some() {
            for (i, vue) in vue_a.iter().enumerate() {
                etat.depuis[i] = match vue {
                    Some(m) => mesures - m,
                    // Jamais vue : aussi ancienne que le journal (elle sera de
                    // toute façon active, son dernier score étant None).
                    None => mesures + 1,
                };
                etat.actives[i] = *vue == Some(mesures) && !vue_en_resondage[i];
            }
        }
        etat
    }

    /// Plan de la prochaine mesure pour un budget TOTAL de `total_parties`.
    pub fn plan(&self, ancres: &[Ancre], total_parties: usize) -> Vec<PlanAncre> {
        plan_adaptatif_eligibles(
            ancres,
            &self.derniers,
            &self.depuis,
            &self.actives,
            &vec![true; ancres.len()],
            total_parties,
        )
    }

    /// Idem, restreint aux ancres jouables dans le contexte courant (voir
    /// `plan_adaptatif_eligibles` : ancres UCI sans moteur, régime 1-pli).
    pub fn plan_eligibles(&self, ancres: &[Ancre], eligibles: &[bool],
                          total_parties: usize) -> Vec<PlanAncre> {
        plan_adaptatif_eligibles(ancres, &self.derniers, &self.depuis, &self.actives,
                                 eligibles, total_parties)
    }

    /// Enregistre les scores d'une mesure : les ancres jouées rajeunissent et
    /// prennent leur nouveau score, les autres vieillissent d'une mesure.
    /// Toutes les ancres jouées sont réputées ACTIVES (voir `enregistre_plan`
    /// pour distinguer les re-sondages).
    pub fn enregistre(&mut self, ancres: &[Ancre], mesures: &[MesureAncre]) {
        self.enregistre_plan(ancres, mesures, &[]);
    }

    /// Idem, en distinguant les ancres jouées comme ACTIVES de celles jouées au
    /// seul titre du re-sondage : seules les premières gagnent le bénéfice de la
    /// bande de maintien à la mesure suivante (voir `SCORE_MAINTIEN_MIN`). Une
    /// ancre re-sondée reste officiellement saturée : pour reprendre du budget,
    /// il lui faut franchir la bande d'ENTRÉE, la plus exigeante — c'est ce qui
    /// empêche un sondage à faible volume, donc bruité, de la ré-installer dans
    /// le plan sur un coup de chance. `plan` vide = toutes actives.
    pub fn enregistre_plan(&mut self, ancres: &[Ancre], mesures: &[MesureAncre],
                           plan: &[PlanAncre]) {
        for (i, a) in ancres.iter().enumerate() {
            let resondee = plan.iter().any(|e| e.index == i && e.resondage);
            match mesures.iter().find(|m| m.nom == a.nom) {
                Some(m) => {
                    self.derniers[i] = Some(m.score);
                    self.depuis[i] = 0;
                    self.actives[i] = !resondee;
                }
                None => {
                    self.depuis[i] = self.depuis[i].saturating_add(1);
                    self.actives[i] = false;
                }
            }
        }
    }

    /// Derniers scores connus (diagnostic et tests).
    pub fn derniers(&self) -> &[Option<f64>] {
        &self.derniers
    }

    /// Ancienneté, en mesures, du dernier duel de chaque ancre (idem).
    pub fn depuis(&self) -> &[u64] {
        &self.depuis
    }

    /// Ancres actives à la dernière mesure (idem) : celles qui bénéficient de la
    /// bande de maintien de l'hystérésis.
    pub fn actives(&self) -> &[bool] {
        &self.actives
    }
}

/// Bot réseau POSSÉDANT son modèle (Arc), pour satisfaire le 'static exigé par
/// les fabriques d'arena::score. Chaque coup délègue à un NetBot frais semé par
/// le RNG de la partie (même astuce que l'entraîneur).
struct BotReseau {
    net: Arc<Mlp>,
    rng: StdRng,
    depth: u32,
}

impl BotReseau {
    fn new(net: Arc<Mlp>, graine: u64, depth: u32) -> Self {
        BotReseau { net, rng: StdRng::seed_from_u64(graine), depth }
    }
}

impl Bot for BotReseau {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        let graine_coup: u64 = self.rng.gen();
        NetBot::new(&self.net, graine_coup, 0.0, self.depth).choose(pos)
    }
}

/// Joue `parties_par_ancre` parties contre CHAQUE ancre maison (échelle
/// complète, comportement historique). Les ancres UCI de la liste sont
/// ignorées ici (elles exigent un chemin moteur : voir mesure_uci).
pub fn mesure(net: &Arc<Mlp>, depth: u32, parties_par_ancre: usize,
              graine: u64) -> Vec<MesureAncre> {
    mesure_plan(net, depth, &plan_complet(ANCRES, parties_par_ancre), graine)
}

/// Joue les ancres MAISON du `plan` (parallélisées par arena::score) et renvoie
/// les scores mesurés — chaque ancre avec le volume de parties que le plan lui
/// alloue. Les ancres UCI du plan sont ignorées ici (voir mesure_uci_plan).
/// La graine de chaque duel dérive du RANG DANS LE GENRE de l'ancre, pas de sa
/// position dans le plan : les duels restent bit à bit ceux de l'historique,
/// qu'une ancre voisine soit jouée ou non.
pub fn mesure_plan(net: &Arc<Mlp>, depth: u32, plan: &[PlanAncre],
                   graine: u64) -> Vec<MesureAncre> {
    plan.iter()
        .filter(|e| e.index < ANCRES.len() && e.parties > 0)
        .filter_map(|e| match ANCRES[e.index].genre {
            GenreAncre::Maison { profondeur } => Some((e, &ANCRES[e.index], profondeur)),
            GenreAncre::Uci { .. } => None,
        })
        .map(|(e, a, profondeur)| {
            let k = rang_dans_genre(ANCRES, e.index);
            let net_a = net.clone();
            let score = arena::score(
                move |g: u64| -> Box<dyn Bot> {
                    Box::new(BotReseau::new(net_a.clone(), g, depth))
                },
                |g: u64| -> Box<dyn Bot> {
                    match profondeur {
                        None => Box::new(RandomBot::new(g)),
                        Some(d) => Box::new(MaterialBot::new(g, d)),
                    }
                },
                e.parties,
                graine.wrapping_add(k as u64).wrapping_mul(0x9E37_79B9),
            ) as f64;
            MesureAncre { nom: a.nom, elo_ancre: a.elo, score, parties: e.parties }
        })
        .collect()
}

/// Joue les ancres UCI de l'échelle contre l'agent fabriqué par `fabrique`
/// (une instance moteur PAR PARTIE, indispensable au parallélisme
/// d'arena::score — même pattern que calibrate.rs). Dégradation gracieuse :
/// chemin vide, moteur introuvable ou duel en échec → ancre(s) sautée(s) avec
/// message clair et JAMAIS de panique ; le fit retombe sur les ancres maison.
pub fn mesure_uci<F>(fabrique: F, chemin_moteur: &str, parties_par_ancre: usize,
                     graine: u64) -> Vec<MesureAncre>
where
    F: Fn(u64) -> Box<dyn Bot> + Sync,
{
    mesure_uci_liste(fabrique, chemin_moteur, ANCRES, parties_par_ancre, graine)
}

/// Joue les ancres UCI du `plan`, chacune avec son volume de parties (jumeau
/// UCI de `mesure_plan`). Mêmes dégradations gracieuses que `mesure_uci`.
pub fn mesure_uci_plan<F>(fabrique: F, chemin_moteur: &str, plan: &[PlanAncre],
                          graine: u64) -> Vec<MesureAncre>
where
    F: Fn(u64) -> Box<dyn Bot> + Sync,
{
    mesure_uci_plan_liste(fabrique, chemin_moteur, ANCRES, plan, graine)
}

/// Cœur de `mesure_uci`, liste d'ancres paramétrable (testabilité : les tests
/// jouent une ancre courte au lieu des 2 × 24 parties de production).
fn mesure_uci_liste<F>(fabrique: F, chemin_moteur: &str, ancres: &[Ancre],
                       parties_par_ancre: usize, graine: u64) -> Vec<MesureAncre>
where
    F: Fn(u64) -> Box<dyn Bot> + Sync,
{
    let plan = plan_complet(ancres, parties_par_ancre);
    mesure_uci_plan_liste(fabrique, chemin_moteur, ancres, &plan, graine)
}

/// Cœur commun : ancres UCI du plan, liste d'ancres paramétrable.
fn mesure_uci_plan_liste<F>(fabrique: F, chemin_moteur: &str, ancres: &[Ancre],
                            plan: &[PlanAncre], graine: u64) -> Vec<MesureAncre>
where
    F: Fn(u64) -> Box<dyn Bot> + Sync,
{
    // Retenues du plan, dans l'ordre de l'échelle.
    let retenues: Vec<&PlanAncre> = plan
        .iter()
        .filter(|e| {
            e.index < ancres.len()
                && e.parties > 0
                && matches!(ancres[e.index].genre, GenreAncre::Uci { .. })
        })
        .collect();
    // Aucune ancre UCI demandée (échelle adaptative : elles peuvent toutes être
    // saturées) → on ne lance même pas la sonde moteur. Une seule exception au
    // silence : SANS moteur, les ancres UCI n'ont pas été « écartées » par le
    // plan, elles étaient injouables — l'opérateur cherche ce repère historique
    // dans le journal, et le taire ferait passer une absence d'outil pour une
    // saturation.
    if retenues.is_empty() {
        let echelle_a_de_l_uci =
            ancres.iter().any(|a| matches!(a.genre, GenreAncre::Uci { .. }));
        if chemin_moteur.is_empty() && echelle_a_de_l_uci {
            println!(
                "  echelle Elo : ancres UCI sautees (aucun moteur fourni, voir --oracle)"
            );
        }
        return Vec::new();
    }
    if chemin_moteur.is_empty() {
        println!("  echelle Elo : ancres UCI sautees (aucun moteur fourni, voir --oracle)");
        return Vec::new();
    }
    // Sonde : vérifie le binaire et relève les bornes UCI_Elo AVANT de jouer
    // (même préambule que calibrate.rs) ; Drop = quit propre, pas de zombie.
    let (elo_min, elo_max) = match UciEngine::lance(chemin_moteur) {
        Ok(sonde) => (sonde.elo_min, sonde.elo_max),
        Err(e) => {
            println!("  echelle Elo : ancres UCI sautees ({chemin_moteur} : {e})");
            return Vec::new();
        }
    };
    retenues
        .into_iter()
        .filter_map(|e| {
            let a = &ancres[e.index];
            let GenreAncre::Uci { elo_nominal, movetime_ms } = a.genre else {
                return None; // impossible : `retenues` ne garde que l'UCI
            };
            // Clamp aux bornes annoncées (Stockfish : 1320..3190) ; l'Elo
            // EFFECTIVEMENT appliqué alimente le fit, pas le nominal.
            let elo_reel = elo_nominal.clamp(elo_min, elo_max);
            if elo_reel != elo_nominal {
                println!("  echelle Elo : {} clampe a UCI_Elo {elo_reel}", a.nom);
            }
            // Graine dérivée du RANG DANS LE GENRE (et non de la position dans
            // le plan) : les duels d'une ancre ne bougent pas quand l'échelle
            // adaptative en écarte une autre.
            let k = rang_dans_genre(ancres, e.index);
            let parties = e.parties;
            // catch_unwind : la sonde a validé le binaire, mais un lancement
            // qui échoue EN COURS de duel (épuisement de ressources, moteur
            // supprimé entre-temps) paniquerait dans la fabrique adverse —
            // mieux vaut sauter l'ancre que tuer la mesure (et la nuit
            // d'entraînement avec elle).
            let resultat = catch_unwind(AssertUnwindSafe(|| {
                arena::score(
                    &fabrique,
                    |_g: u64| -> Box<dyn Bot> {
                        Box::new(
                            StockfishBot::new(chemin_moteur, elo_reel, movetime_ms)
                                .unwrap_or_else(|e| panic!("lancement du moteur bride : {e}")),
                        )
                    },
                    parties,
                    graine.wrapping_add(0x5F00 + k as u64).wrapping_mul(0x9E37_79B9),
                ) as f64
            }));
            match resultat {
                Ok(score) => {
                    println!(
                        "  echelle Elo : {} -> {:.0} % ({} parties{})",
                        a.nom,
                        score * 100.0,
                        parties,
                        if e.resondage { ", re-sondage" } else { "" }
                    );
                    std::io::stdout().flush().ok();
                    Some(MesureAncre {
                        nom: a.nom,
                        elo_ancre: elo_reel as f64,
                        score,
                        parties,
                    })
                }
                Err(_) => {
                    println!("  echelle Elo : ancre {} sautee (duel UCI en echec)", a.nom);
                    None
                }
            }
        })
        .collect()
}

/// Ajuste l'Elo par maximum de vraisemblance binomiale sur toutes les mesures.
/// Les scores extrêmes sont adoucis (un 100 % sur n parties vaut « au plus
/// 1 - 1/(2n) ») pour garder la vraisemblance finie ; la log-vraisemblance est
/// unimodale en R → recherche ternaire sur [0, 3200].
///
/// AUCUNE hypothèse sur le nombre d'ancres ni sur l'égalité des tailles
/// d'échantillon : chaque mesure entre dans la somme avec SON `parties` (le
/// terme est n·[s·ln p + (1-s)·ln(1-p)], donc pondéré par le volume) et
/// l'adoucissement des extrêmes se calcule ancre par ancre. C'est le point de
/// contrat qui rend l'échelle adaptative légitime : un plan à 3 ancres de
/// 56 parties et un plan à 7 ancres de 24 parties se traitent pareil.
pub fn ajuste_elo(mesures: &[MesureAncre]) -> f64 {
    let ll = |r: f64| -> f64 {
        mesures
            .iter()
            .map(|m| {
                let n = m.parties as f64;
                let s = m.score.clamp(0.5 / n, 1.0 - 0.5 / n);
                let p = 1.0 / (1.0 + 10f64.powf((m.elo_ancre - r) / 400.0));
                n * (s * p.ln() + (1.0 - s) * (1.0 - p).ln())
            })
            .sum()
    };
    let (mut lo, mut hi) = (0.0f64, 3200.0f64);
    for _ in 0..90 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        if ll(m1) < ll(m2) {
            lo = m1;
        } else {
            hi = m2;
        }
    }
    (lo + hi) / 2.0
}

/// Elo IMPLIQUÉ par une mesure prise SEULE : l'inverse de la logistique, avec le
/// même adoucissement des extrêmes que `ajuste_elo`.
///
/// Sous un modèle bien spécifié, toutes les ancres d'une même mesure impliquent
/// le même Elo. Sur le journal réel elles s'écartent de plus de 200 points
/// (materiel d3 1796 contre stockfish 1700 1573 sur les 40 dernières mesures) :
/// les Elo d'ancres sont mal calibrés ENTRE EUX, et le MLE ne fait alors que
/// projeter ce désaccord — la valeur publiée dépend donc de l'ensemble d'ancres
/// jouées, pas seulement de la force du réseau. D'où `dispersion_ancres`.
pub fn elo_implique(m: &MesureAncre) -> f64 {
    let n = (m.parties.max(1)) as f64;
    let s = m.score.clamp(0.5 / n, 1.0 - 0.5 / n);
    m.elo_ancre + 400.0 * (s / (1.0 - s)).log10()
}

/// Écart-type des Elo impliqués ancre par ancre : mesure d'ADÉQUATION du fit.
///
/// C'est la statistique qui manquait — le modèle n'ayant qu'un paramètre, rien
/// ne signalait que les ancres se contredisaient. Une dispersion qui enfle est
/// le symptôme d'une échelle d'ancres à recalibrer, pas d'un réseau qui bouge ;
/// journalisée à chaque mesure, elle transforme une échelle silencieusement
/// fausse en alarme visible. Moins de deux ancres → 0 (rien à comparer).
pub fn dispersion_ancres(mesures: &[MesureAncre]) -> f64 {
    if mesures.len() < 2 {
        return 0.0;
    }
    let impliques: Vec<f64> = mesures.iter().map(elo_implique).collect();
    let moyenne = impliques.iter().sum::<f64>() / impliques.len() as f64;
    let variance = impliques.iter().map(|e| (e - moyenne).powi(2)).sum::<f64>()
        / impliques.len() as f64;
    variance.sqrt()
}

/// Une mesure porte-t-elle une information INTÉRIEURE, c'est-à-dire au moins un
/// score qui ne soit pas collé à 0 % ou 100 % ?
///
/// Si toutes les ancres jouées sortent à 100 % (ou 0 %), chaque terme de la
/// vraisemblance est écrasé sur son adoucissement et l'Elo ajusté devient une
/// fonction DÉTERMINISTE des Elo d'ancres et du nombre de parties : plus aucune
/// information ne vient des duels, et le résultat bouge avec le seul volume —
/// un 100 % implique ancre + 417 à 6 parties, + 669 à 24, + 818 à 56. Changer la
/// taille d'échantillon suffisait alors à faire bondir la courbe de plus de
/// 200 points sans que le réseau ait joué autrement. Un tel point ne doit pas
/// être publié : c'est le rôle de ce test au site d'appel.
///
/// Critère : au moins une ancre à au moins un point entier des bornes
/// (`1/n <= s <= 1 - 1/n`), donc porteuse d'un vrai gradient.
pub fn mesure_informative(mesures: &[MesureAncre]) -> bool {
    mesures.iter().any(|m| {
        let n = (m.parties.max(1)) as f64;
        m.score >= 1.0 / n && m.score <= 1.0 - 1.0 / n
    })
}

/// Empreinte compacte d'un ensemble d'ancres : bits d'index (ancre 0 → bit 0).
/// Journalisée avec chaque point d'Elo pour que la courbe se décrive elle-même
/// — l'ensemble actif changeant d'une mesure à l'autre, un décrochage doit
/// pouvoir être imputé au plan plutôt qu'au réseau. Au-delà de 32 ancres, les
/// bits hauts sont ignorés (l'échelle en compte 8).
pub fn empreinte_ancres(ancres: &[Ancre], mesures: &[MesureAncre]) -> u32 {
    let mut bits = 0u32;
    for m in mesures {
        if let Some(i) = ancres.iter().position(|a| a.nom == m.nom) {
            if i < 32 {
                bits |= 1 << i;
            }
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores synthétiques générés par la logistique d'un « vrai » Elo sur
    /// l'échelle MIXTE (maison + UCI) : l'ajustement doit le retrouver à
    /// quelques points près, y compris au-dessus des ancres maison (1800).
    #[test]
    fn retrouve_un_elo_connu() {
        for vrai in [600.0, 1000.0, 1400.0, 1800.0] {
            let mesures: Vec<MesureAncre> = ANCRES
                .iter()
                .map(|a| MesureAncre {
                    nom: a.nom,
                    elo_ancre: a.elo,
                    score: 1.0 / (1.0 + 10f64.powf((a.elo - vrai) / 400.0)),
                    parties: 1000,
                })
                .collect();
            let estime = ajuste_elo(&mesures);
            assert!(
                (estime - vrai).abs() < 15.0,
                "vrai {vrai}, estimé {estime}"
            );
        }
    }

    /// La raison d'être des ancres UCI : un agent à 1750 écrase TOUTES les
    /// ancres maison (saturation) mais tient des scores intermédiaires contre
    /// stockfish 1700/2000 — le fit mixte doit retomber près de 1750 là où le
    /// fit maison seul n'a plus de résolution.
    #[test]
    fn ancres_mixtes_desaturent_le_haut() {
        let vrai = 1750.0;
        let logistique =
            |elo_ancre: f64| 1.0 / (1.0 + 10f64.powf((elo_ancre - vrai) / 400.0));
        let mixtes: Vec<MesureAncre> = ANCRES
            .iter()
            .map(|a| MesureAncre {
                nom: a.nom,
                elo_ancre: a.elo,
                score: logistique(a.elo),
                parties: 24,
            })
            .collect();
        let estime = ajuste_elo(&mixtes);
        assert!((estime - vrai).abs() < 30.0, "vrai {vrai}, estimé {estime}");
    }

    /// Tout écraser (100 % partout) doit donner une estimation au-dessus de la
    /// plus haute ancre (2000 désormais), sans diverger.
    #[test]
    fn score_parfait_reste_borne() {
        let mesures: Vec<MesureAncre> = ANCRES
            .iter()
            .map(|a| MesureAncre { nom: a.nom, elo_ancre: a.elo, score: 1.0, parties: 24 })
            .collect();
        let estime = ajuste_elo(&mesures);
        assert!(estime > 2000.0 && estime <= 3200.0, "estimé {estime}");
    }

    /// Dégradation gracieuse : chemin vide ou binaire introuvable → aucune
    /// mesure UCI, aucun panic (le fit retombe sur les ancres maison).
    #[test]
    fn moteur_absent_saute_les_ancres_uci() {
        let fabrique = |g: u64| -> Box<dyn Bot> { Box::new(RandomBot::new(g)) };
        assert!(mesure_uci(fabrique, "", 2, 0).is_empty());
        let fabrique = |g: u64| -> Box<dyn Bot> { Box::new(RandomBot::new(g)) };
        assert!(mesure_uci(fabrique, "moteur/inexistant.exe", 2, 0).is_empty());
    }

    // --- Échelle adaptative (R2) ---------------------------------------------

    /// Somme des parties d'un plan.
    fn total(plan: &[PlanAncre]) -> usize {
        plan.iter().map(|e| e.parties).sum()
    }

    /// Indices des ancres retenues par un plan, dans l'ordre.
    fn indices(plan: &[PlanAncre]) -> Vec<usize> {
        plan.iter().map(|e| e.index).collect()
    }

    /// Budget de production : 7 ancres × 24 parties de l'échelle historique.
    const BUDGET: usize = 168;

    /// PREMIER DÉMARRAGE (aucun historique) : toutes les ancres sont jouées,
    /// comme avant ce chantier — et le budget total est respecté au pli près.
    #[test]
    fn premier_demarrage_joue_toutes_les_ancres() {
        let n = ANCRES.len();
        let plan = plan_adaptatif(ANCRES, &vec![None; n], &vec![0; n], BUDGET);
        assert_eq!(indices(&plan), (0..n).collect::<Vec<_>>());
        assert_eq!(total(&plan), BUDGET);
        assert!(plan.iter().all(|e| e.parties % 2 == 0), "volumes pairs (couleurs)");
        assert!(plan.iter().all(|e| !e.resondage));
    }

    /// Les ancres SATURÉES (hors [15 %, 85 %]) sont écartées, et tout le budget
    /// va aux informatives — une ancre jamais mesurée (la nouvelle 2300) est
    /// considérée informative tant qu'on ne l'a pas sondée.
    #[test]
    fn ancres_saturees_ecartees_et_budget_redistribue() {
        let derniers = vec![
            Some(1.0),  // aleatoire      : saturée
            Some(0.99), // materiel d1    : saturée
            Some(0.95), // materiel d2    : saturée
            Some(0.90), // materiel d3    : saturée
            Some(0.70), // materiel d4    : informative
            Some(0.40), // stockfish 1700 : informative
            Some(0.19), // stockfish 2000 : informative
            None,       // stockfish 2300 : jamais sondée
        ];
        let plan = plan_adaptatif(ANCRES, &derniers, &vec![0; ANCRES.len()], BUDGET);
        assert_eq!(indices(&plan), vec![4, 5, 6, 7]);
        assert_eq!(total(&plan), BUDGET);
        // 168 parties pour 4 ancres actives : 42 chacune (contre 24 avant).
        assert!(plan.iter().all(|e| e.parties == 42), "{plan:?}");
    }

    /// PLANCHER : même quand TOUTES les ancres sont saturées, trois au moins
    /// sont jouées — et ce sont les MOINS éloignées de la plage informative.
    #[test]
    fn plancher_de_trois_ancres_les_moins_saturees() {
        let derniers = vec![
            Some(1.00), // 0 : distance 0.15
            Some(1.00), // 1 : distance 0.15
            Some(0.95), // 2 : distance 0.10
            Some(0.90), // 3 : distance 0.05
            Some(0.88), // 4 : distance 0.03  <- la plus proche
            Some(0.90), // 5 : distance 0.05
            Some(0.95), // 6 : distance 0.10
            Some(1.00), // 7 : distance 0.15
        ];
        let plan = plan_adaptatif(ANCRES, &derniers, &vec![0; ANCRES.len()], BUDGET);
        assert_eq!(indices(&plan), vec![3, 4, 5], "les trois moins saturées");
        assert_eq!(total(&plan), BUDGET);
        assert!(plan.iter().all(|e| e.parties == 56), "168 / 3 = 56 : {plan:?}");
    }

    /// RE-SONDAGE : une ancre saturée depuis longtemps est rejouée à faible
    /// volume (6 parties), prélevé sur le budget — le total reste celui demandé.
    #[test]
    fn resondage_periodique_des_ancres_saturees() {
        let derniers = vec![
            Some(1.0),
            Some(1.0),
            Some(0.5),
            Some(0.5),
            Some(0.5),
            Some(0.9),
            Some(0.95),
            Some(1.0),
        ];
        let mut depuis = vec![0u64; ANCRES.len()];
        depuis[0] = echeance_resondage(0) + 10; // saturée ET oubliée depuis longtemps
        depuis[1] = echeance_resondage(1) - 1; // saturée mais pas encore à échéance
        let plan = plan_adaptatif(ANCRES, &derniers, &depuis, BUDGET);
        assert_eq!(indices(&plan), vec![0, 2, 3, 4]);
        let resondee = plan.iter().find(|e| e.index == 0).expect("ancre re-sondée");
        assert!(resondee.resondage);
        assert_eq!(resondee.parties, PARTIES_RESONDAGE);
        assert_eq!(total(&plan), BUDGET, "le re-sondage se prélève sur le budget");
        // Les trois actives se partagent 168 - 12 = 156 parties, par paires.
        assert!(plan.iter().filter(|e| e.index != 0).all(|e| e.parties == 52), "{plan:?}");
    }

    /// Les échéances de re-sondage sont DÉSYNCHRONISÉES : cinq ancres saturées à
    /// la même mesure ne doivent pas revenir toutes ensemble et prendre d'un
    /// coup 5 × 12 parties sur le budget.
    #[test]
    fn echeances_de_resondage_desynchronisees() {
        let echeances: Vec<u64> = (0..ANCRES.len()).map(echeance_resondage).collect();
        let mut uniques = echeances.clone();
        uniques.sort_unstable();
        uniques.dedup();
        assert_eq!(uniques.len(), echeances.len(), "échéances confondues : {echeances:?}");
        assert!(echeances.iter().all(|&e| e >= PERIODE_RESONDAGE));
        // Cas concret : cinq ancres maison saturées EN PHASE (même `depuis`).
        let derniers = vec![Some(1.0); ANCRES.len()];
        let depuis = vec![PERIODE_RESONDAGE + 1; ANCRES.len()];
        let plan = plan_adaptatif(ANCRES, &derniers, &depuis, BUDGET);
        let rafale = plan.iter().filter(|e| e.resondage).count();
        assert!(rafale <= 1, "rafale de {rafale} re-sondages dans la même mesure : {plan:?}");
    }

    /// HYSTÉRÉSIS : une ancre DÉJÀ active survit à un score qui déborde la bande
    /// d'entrée sans sortir de la bande de maintien — c'est le cas réel de
    /// materiel d4, sortie du plan à 85,4 % (0,4 point au-dessus du couperet)
    /// alors qu'elle est l'ancre la plus informative de l'échelle.
    #[test]
    fn hysteresis_garde_une_ancre_active_a_85_pourcent() {
        let n = ANCRES.len();
        // État réel du journal à h238.119.
        let derniers = vec![
            Some(1.000), // aleatoire
            Some(0.958), // materiel d1
            Some(0.979), // materiel d2
            Some(0.917), // materiel d3
            Some(0.854), // materiel d4  <- au-dessus de 0.85, sous 0.90
            Some(0.354), // stockfish 1700
            Some(0.188), // stockfish 2000
            None,        // stockfish 2300
        ];
        let depuis = vec![0u64; n];
        // Toutes ces ancres viennent d'être jouées comme ACTIVES.
        let actives = vec![true; n];
        let plan = plan_adaptatif_eligibles(
            ANCRES, &derniers, &depuis, &actives, &vec![true; n], BUDGET,
        );
        assert!(indices(&plan).contains(&4), "d4 doit rester active : {plan:?}");
        assert_eq!(indices(&plan), vec![4, 5, 6, 7]);
        // Sans le bénéfice de l'hystérésis (ancre inactive qui DEMANDE à entrer),
        // le même score de 85,4 % ne suffit pas : la bande d'entrée est fermée.
        let plan_entree = plan_adaptatif_eligibles(
            ANCRES, &derniers, &depuis, &vec![false; n], &vec![true; n], BUDGET,
        );
        assert!(!indices(&plan_entree).contains(&4), "{plan_entree:?}");
        // Franchement saturée (au-delà de la bande de maintien) : écartée même
        // en étant active — l'hystérésis élargit la bande, elle ne la supprime pas.
        let mut franche = derniers.clone();
        franche[4] = Some(0.95);
        let plan_saturee = plan_adaptatif_eligibles(
            ANCRES, &franche, &depuis, &actives, &vec![true; n], BUDGET,
        );
        assert!(!indices(&plan_saturee).contains(&4), "{plan_saturee:?}");
    }

    /// Le drapeau d'activité suit le PLAN : une ancre jouée en re-sondage reste
    /// officiellement saturée (bande d'entrée à la mesure suivante), une ancre
    /// jouée à plein volume devient active (bande de maintien).
    #[test]
    fn resondage_ne_rend_pas_une_ancre_active() {
        let mut etat = EtatEchelle::neuf(ANCRES.len());
        let plan = vec![
            PlanAncre { index: 0, parties: PARTIES_RESONDAGE, resondage: true },
            PlanAncre { index: 5, parties: 56, resondage: false },
        ];
        let mesures = vec![
            MesureAncre { nom: ANCRES[0].nom, elo_ancre: ANCRES[0].elo, score: 0.88, parties: PARTIES_RESONDAGE },
            MesureAncre { nom: ANCRES[5].nom, elo_ancre: ANCRES[5].elo, score: 0.88, parties: 56 },
        ];
        etat.enregistre_plan(ANCRES, &mesures, &plan);
        assert!(!etat.actives()[0], "un re-sondage ne réactive pas une ancre saturée");
        assert!(etat.actives()[5], "une ancre jouée à plein volume est active");
        // Conséquence sur le plan suivant : à 88 %, seule l'ancre 5 est gardée.
        let plan_suivant = etat.plan(ANCRES, BUDGET);
        assert!(!indices(&plan_suivant).contains(&0), "{plan_suivant:?}");
        assert!(indices(&plan_suivant).contains(&5), "{plan_suivant:?}");
    }

    /// ANCRES INÉLIGIBLES : l'état réel du journal (toutes les ancres maison
    /// saturées, seules les Stockfish informatives) donnerait une mesure VIDE
    /// sans moteur UCI. Restreint aux ancres maison, le plan retombe sur les
    /// trois moins saturées et dépense tout le budget — jamais de mesure creuse.
    #[test]
    fn ancres_ineligibles_ecartees_avant_selection() {
        let derniers = vec![
            Some(1.00), // aleatoire
            Some(0.96), // materiel d1
            Some(0.98), // materiel d2
            Some(0.92), // materiel d3
            Some(0.85), // materiel d4 (pile à la borne : informative)
            Some(0.35), // stockfish 1700
            Some(0.19), // stockfish 2000
            None,       // stockfish 2300
        ];
        let depuis = vec![0u64; ANCRES.len()];
        // Sans moteur : seules les ancres maison sont jouables.
        let maison: Vec<bool> = ANCRES
            .iter()
            .map(|a| matches!(a.genre, GenreAncre::Maison { .. }))
            .collect();
        let plan = plan_adaptatif_eligibles(
            ANCRES, &derniers, &depuis, &[], &maison, BUDGET,
        );
        assert!(!plan.is_empty(), "une mesure sans moteur doit rester possible");
        assert!(
            plan.iter().all(|e| matches!(ANCRES[e.index].genre, GenreAncre::Maison { .. })),
            "aucune ancre UCI ne doit être planifiée sans moteur : {plan:?}"
        );
        assert_eq!(total(&plan), BUDGET, "tout le budget est dépensé");
        // d4 (85 %, pile dans la plage) est active ; le plancher la complète
        // par les deux moins saturées : d3 (0.92) puis d1 (0.96).
        assert_eq!(indices(&plan), vec![1, 3, 4]);
        // Avec moteur, la même mémoire préfère les ancres hautes.
        let toutes = plan_adaptatif(ANCRES, &derniers, &depuis, BUDGET);
        assert_eq!(indices(&toutes), vec![4, 5, 6, 7]);
    }

    /// L'échelle adaptative ne doit JAMAIS décaler les duels d'une ancre selon
    /// que ses voisines sont jouées ou non : la graine dérive du rang dans le
    /// genre, invariant du plan.
    #[test]
    fn rang_dans_genre_invariant() {
        assert_eq!(rang_dans_genre(ANCRES, 0), 0); // aleatoire
        assert_eq!(rang_dans_genre(ANCRES, 4), 4); // materiel d4
        assert_eq!(rang_dans_genre(ANCRES, 5), 0); // stockfish 1700 (1re UCI)
        assert_eq!(rang_dans_genre(ANCRES, 6), 1); // stockfish 2000
        assert_eq!(rang_dans_genre(ANCRES, 7), 2); // stockfish 2300 (ajoutée en fin)
    }

    /// Mémoire de l'échelle : une mesure partielle rajeunit les ancres jouées
    /// et vieillit les autres — c'est ce compteur qui déclenche le re-sondage.
    #[test]
    fn etat_echelle_vieillit_les_ancres_non_jouees() {
        let mut etat = EtatEchelle::neuf(ANCRES.len());
        let mesures = vec![MesureAncre {
            nom: ANCRES[5].nom,
            elo_ancre: ANCRES[5].elo,
            score: 0.42,
            parties: 56,
        }];
        etat.enregistre(ANCRES, &mesures);
        assert_eq!(etat.derniers()[5], Some(0.42));
        assert_eq!(etat.depuis()[5], 0);
        assert_eq!(etat.derniers()[0], None);
        assert_eq!(etat.depuis()[0], 1);
        etat.enregistre(ANCRES, &[]);
        assert_eq!(etat.depuis()[0], 2);
        assert_eq!(etat.depuis()[5], 1);
    }

    /// Reconstruction depuis un journal ancres.csv : dernier score par ancre et
    /// ancienneté en MESURES (les lignes d'une mesure partagent leurs heures).
    /// Journal absent → état vierge (échelle complète).
    #[test]
    fn etat_echelle_relu_du_journal() {
        let dossier = std::env::temp_dir().join("echec_test_ancres_csv");
        std::fs::create_dir_all(&dossier).expect("dossier temporaire");
        let chemin = dossier.join("ancres.csv");
        std::fs::write(
            &chemin,
            "heures,ancre,score_pct,parties\n\
             10.000,aleatoire,100.0,24\n\
             10.000,materiel d4,60.0,24\n\
             10.000,stockfish 1700,40.0,24\n\
             11.000,materiel d4,70.0,56\n\
             11.000,stockfish 1700,45.0,56\n\
             12.000,materiel d4,72.0,56\n\
             12.000,stockfish 1700,47.0,56\n",
        )
        .expect("écriture du journal de test");
        let etat = EtatEchelle::charge_csv(chemin.to_str().unwrap(), ANCRES);
        assert_eq!(etat.derniers()[0], Some(1.0), "aleatoire : dernier score connu");
        assert_eq!(etat.depuis()[0], 2, "absente des deux dernières mesures");
        assert_eq!(etat.derniers()[4], Some(0.72));
        assert_eq!(etat.depuis()[4], 0);
        assert_eq!(etat.derniers()[7], None, "stockfish 2300 : jamais mesurée");
        // Le plan qui en découle : aleatoire est saturée et écartée.
        let plan = etat.plan(ANCRES, BUDGET);
        assert!(!indices(&plan).contains(&0), "ancre saturée écartée : {plan:?}");
        assert!(indices(&plan).contains(&7), "ancre neuve sondée : {plan:?}");
        std::fs::remove_file(&chemin).ok();
        // Journal absent : état vierge, échelle complète.
        let vierge = EtatEchelle::charge_csv(chemin.to_str().unwrap(), ANCRES);
        assert_eq!(vierge.plan(ANCRES, BUDGET).len(), ANCRES.len());
    }

    /// L'ajustement MLE tolère un nombre VARIABLE d'ancres et des tailles
    /// d'échantillon différentes : même Elo retrouvé avec l'échelle complète à
    /// 24 parties et avec une échelle réduite à trois ancres de 56 parties.
    #[test]
    fn ajuste_elo_tolere_un_nombre_variable_d_ancres() {
        let vrai = 1900.0;
        let logistique = |e: f64| 1.0 / (1.0 + 10f64.powf((e - vrai) / 400.0));
        let synthetique = |indices: &[usize], parties: usize| -> Vec<MesureAncre> {
            indices
                .iter()
                .map(|&i| MesureAncre {
                    nom: ANCRES[i].nom,
                    elo_ancre: ANCRES[i].elo,
                    score: logistique(ANCRES[i].elo),
                    parties,
                })
                .collect()
        };
        let complete = synthetique(&(0..ANCRES.len()).collect::<Vec<_>>(), 24);
        let reduite = synthetique(&[5, 6, 7], 56);
        // Tailles d'échantillon MIXTES dans une même liste (cas du re-sondage).
        let mut mixte = synthetique(&[5, 6, 7], 54);
        mixte.extend(synthetique(&[0], 6));
        for (nom, mesures) in
            [("complete", &complete), ("reduite", &reduite), ("mixte", &mixte)]
        {
            let estime = ajuste_elo(mesures);
            assert!((estime - vrai).abs() < 30.0, "{nom} : vrai {vrai}, estimé {estime}");
        }
        // Une seule ancre reste exploitable (dégénéré mais borné).
        let seule = ajuste_elo(&synthetique(&[6], 56));
        assert!(seule > 1500.0 && seule < 3200.0, "une ancre : {seule}");
    }

    /// GARDE-FOU du fit dégénéré : quand toutes les ancres jouées sortent à
    /// 100 %, l'Elo ajusté ne dépend plus des parties mais du seul VOLUME —
    /// le simple passage de 24 à 56 parties par ancre le gonfle de plus de
    /// 200 points sans que le réseau ait changé. Un tel point ne doit pas être
    /// publié : `mesure_informative` est le test qui l'interdit.
    #[test]
    fn fit_entierement_aux_bornes_detecte() {
        let clamp = |indices: &[usize], parties: usize| -> Vec<MesureAncre> {
            indices
                .iter()
                .map(|&i| MesureAncre {
                    nom: ANCRES[i].nom,
                    elo_ancre: ANCRES[i].elo,
                    score: 1.0,
                    parties,
                })
                .collect()
        };
        let cinq_a_24 = clamp(&[0, 1, 2, 3, 4], 24);
        let trois_a_56 = clamp(&[2, 3, 4], 56);
        assert!(!mesure_informative(&cinq_a_24), "100 % partout : aucune information");
        assert!(!mesure_informative(&trois_a_56));
        // Démonstration de l'artefact que le garde-fou intercepte : le même
        // réseau (aucun duel perdu nulle part) « gagne » plus de 200 Elo par le
        // seul changement de volume.
        let saut = ajuste_elo(&trois_a_56) - ajuste_elo(&cinq_a_24);
        assert!(saut > 150.0, "artefact de volume attendu, mesuré {saut:.0} Elo");
        // Une seule ancre intérieure suffit à rendre la mesure exploitable.
        let mut mixte = clamp(&[0, 1], 24);
        mixte.push(MesureAncre {
            nom: ANCRES[5].nom,
            elo_ancre: ANCRES[5].elo,
            score: 0.40,
            parties: 24,
        });
        assert!(mesure_informative(&mixte));
        // Le critère est « à au moins un point entier des bornes » : 24/24 non,
        // 23/24 non plus (c'est exactement 1 - 1/n... donc oui), 22/24 oui.
        let un_point = |s: f64| {
            mesure_informative(&[MesureAncre {
                nom: ANCRES[0].nom,
                elo_ancre: ANCRES[0].elo,
                score: s,
                parties: 24,
            }])
        };
        assert!(!un_point(1.0));
        assert!(un_point(23.0 / 24.0));
        assert!(!un_point(23.5 / 24.0));
    }

    /// La DISPERSION des Elo impliqués mesure l'adéquation du fit : nulle quand
    /// les ancres s'accordent, grande quand elles se contredisent (le cas réel :
    /// plus de 200 points d'étendue entre ancres non saturées).
    #[test]
    fn dispersion_revele_des_ancres_qui_se_contredisent() {
        let vrai = 1700.0;
        let logistique = |e: f64| 1.0 / (1.0 + 10f64.powf((e - vrai) / 400.0));
        let accord: Vec<MesureAncre> = [4usize, 5, 6]
            .iter()
            .map(|&i| MesureAncre {
                nom: ANCRES[i].nom,
                elo_ancre: ANCRES[i].elo,
                score: logistique(ANCRES[i].elo),
                parties: 56,
            })
            .collect();
        assert!(dispersion_ancres(&accord) < 1.0, "{}", dispersion_ancres(&accord));
        // Journal réel : d4 à 85 %, SF1700 à 35 %, SF2000 à 19 % impliquent des
        // Elo très différents — c'est ce désaccord que la colonne doit montrer.
        let reel = vec![
            MesureAncre { nom: ANCRES[4].nom, elo_ancre: 1550.0, score: 0.854, parties: 24 },
            MesureAncre { nom: ANCRES[5].nom, elo_ancre: 1700.0, score: 0.354, parties: 24 },
            MesureAncre { nom: ANCRES[6].nom, elo_ancre: 2000.0, score: 0.188, parties: 24 },
        ];
        assert!(dispersion_ancres(&reel) > 50.0, "{}", dispersion_ancres(&reel));
        assert_eq!(dispersion_ancres(&reel[..1]), 0.0, "une seule ancre : rien à comparer");
        // Empreinte : bits d'index des ancres du fit (4, 5, 6 → 0b111_0000).
        assert_eq!(empreinte_ancres(ANCRES, &reel), 0b111_0000);
        assert_eq!(empreinte_ancres(ANCRES, &[]), 0);
    }

    /// La 5e colonne `resondage` d'ancres.csv est relue : une ancre dont la
    /// dernière apparition n'était qu'un re-sondage ne repart PAS active (elle
    /// devra franchir la bande d'entrée), et les journaux historiques à
    /// 4 colonnes restent lus comme des mesures pleines.
    #[test]
    fn resondage_relu_du_journal() {
        let dossier = std::env::temp_dir().join("echec_test_ancres_resondage");
        std::fs::create_dir_all(&dossier).expect("dossier temporaire");
        let chemin = dossier.join("ancres.csv");
        std::fs::write(
            &chemin,
            "heures,ancre,score_pct,parties,resondage\n\
             10.000,aleatoire,100.0,24,0\n\
             10.000,materiel d4,86.0,24,0\n\
             11.000,materiel d4,86.0,56,0\n\
             11.000,aleatoire,88.0,12,1\n",
        )
        .expect("écriture du journal de test");
        let etat = EtatEchelle::charge_csv(chemin.to_str().unwrap(), ANCRES);
        assert_eq!(etat.derniers()[0], Some(0.88));
        assert!(!etat.actives()[0], "jouée en re-sondage : reste saturée");
        assert!(etat.actives()[4], "jouée à plein volume : active");
        // 86 % : d4 garde son budget (bande de maintien), l'aléatoire non.
        let plan = etat.plan(ANCRES, BUDGET);
        assert!(indices(&plan).contains(&4), "{plan:?}");
        assert!(!indices(&plan).contains(&0), "{plan:?}");
        std::fs::remove_file(&chemin).ok();
    }

    /// DIAGNOSTIC (harnais opérateur, ignoré par défaut) : lit le journal réel
    /// models/ancres.csv et imprime l'échelle que la prochaine mesure jouerait
    /// — derniers scores, ancres écartées, budget par ancre retenue. À lancer
    /// après un changement de régime pour voir où part le budget de mesure.
    /// `cargo test --lib diag_plan -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn diag_plan_du_journal() {
        const JOURNAL: &str = "models/ancres.csv";
        if !std::path::Path::new(JOURNAL).exists() {
            println!("journal absent ({JOURNAL}) : diagnostic sauté");
            return;
        }
        let etat = EtatEchelle::charge_csv(JOURNAL, ANCRES);
        let plan = etat.plan(ANCRES, 168);
        println!("--- echelle adaptative, budget 168 parties ---");
        for (i, a) in ANCRES.iter().enumerate() {
            let dernier = match etat.derniers()[i] {
                Some(s) => format!("{:.1} %", s * 100.0),
                None => "jamais mesuree".to_string(),
            };
            let verdict = match plan.iter().find(|e| e.index == i) {
                Some(e) if e.resondage => format!("re-sondage {} parties", e.parties),
                Some(e) => format!("ACTIVE, {} parties", e.parties),
                None => "ecartee (saturee)".to_string(),
            };
            println!(
                "  {:<16} dernier {:>16} (il y a {} mesures) -> {}",
                a.nom,
                dernier,
                etat.depuis()[i],
                verdict
            );
        }
        println!("total joue : {} parties", total(&plan));
    }

    /// Ancre UCI réelle : un moteur bridé est lancé, joue un mini-duel contre
    /// le bot aléatoire, et la mesure revient dans [0, 1] avec l'Elo clampé
    /// aux bornes du moteur. Ignoré par défaut (dépend du binaire local) :
    /// `cargo test --lib -- --ignored ancre_uci`.
    #[test]
    #[ignore = "nécessite engines/stockfish en local"]
    fn ancre_uci_bridee_reelle() {
        const CHEMIN: &str = "engines/stockfish/stockfish-windows-x86-64-avx2.exe";
        // Une seule ancre courte (movetime 10 ms, 2 parties) : le test valide
        // le circuit lancement → duel → mesure, pas la précision statistique.
        let ancres = [Ancre {
            nom: "stockfish test",
            elo: 1000.0, // sous la borne basse : le clamp doit s'appliquer
            genre: GenreAncre::Uci { elo_nominal: 1000, movetime_ms: 10 },
        }];
        let fabrique = |g: u64| -> Box<dyn Bot> { Box::new(RandomBot::new(g)) };
        let mesures = mesure_uci_liste(fabrique, CHEMIN, &ancres, 2, 7);
        assert_eq!(mesures.len(), 1, "l'ancre UCI doit être jouée");
        let m = &mesures[0];
        assert_eq!(m.parties, 2);
        assert!((0.0..=1.0).contains(&m.score), "score {}", m.score);
        // Stockfish annonce min 1320 : l'Elo mesuré est l'Elo APPLIQUÉ.
        assert!(m.elo_ancre >= 1320.0, "clamp attendu : {}", m.elo_ancre);
    }
}
