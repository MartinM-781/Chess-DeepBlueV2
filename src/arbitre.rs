//! Arbitrage d'une partie : conversion des scores UCI, perte en centipions,
//! classement d'un coup et phase de la partie — le CERVEAU de src/bin/arbitre.rs.
//!
//! Tout ce qui est calculable sans moteur ni fichier vit ici, dans la
//! bibliothèque, pour être couvert par `cargo test --lib` (les tests d'un
//! `src/bin/*.rs` ne le sont pas). Le binaire ne garde que ce qui touche au
//! monde : piloter Stockfish, surveiller models/match_live.json, écrire le
//! JSON et le CSV.
//!
//! CONVENTIONS, une fois pour toutes :
//! - un score UCI (`ScoreUci`) est du point de vue du CAMP AU TRAIT ; toutes
//!   les évaluations exposées ici (`eval_avant`, `eval_apres`) sont, elles, du
//!   point de vue des BLANCS — c'est la convention de tout le reste du projet
//!   (jauges du direct, v_champion/v_fantome) et la seule lisible à l'écran ;
//! - un mat vaut ±(MAT_CP − n), n = nombre de coups jusqu'au mat : « mat en 1 »
//!   (31 999) est mieux que « mat en 7 » (31 993), et l'ordre reste total ;
//! - une PERTE est toujours ≥ 0 et du point de vue du camp QUI VIENT DE JOUER.

use serde_json::json;
use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::{Chess, EnPassantMode, Position};

use crate::uci::ScoreUci;

/// Valeur d'un mat immédiat, en centipions (échelle du contrat).
pub const MAT_CP: i32 = 32_000;

/// Plafond appliqué aux DEUX évaluations avant de calculer une perte.
///
/// Sans lui, un coup qui passe de « +3 » à « maté en 4 » compterait pour
/// ~35 000 cp de perte et écraserait à lui seul toutes les moyennes du match :
/// la perte moyenne ne dirait plus rien de la qualité du jeu, seulement de la
/// présence d'un mat. 2 000 cp (20 pions) est très au-delà de toute partie
/// encore disputée : au-dessus, la position est gagnée ou perdue, et la
/// GRADUATION de l'écart n'apprend plus rien. Le classement, lui, reste juste :
/// toute bévue à ce niveau dépasse largement le seuil « gaffe » (300).
/// Les évaluations PUBLIÉES ne sont pas plafonnées : on lit bien « mat en 4 ».
pub const PLAFOND_PERTE: i32 = 2_000;

/// Seuils de classement, en centipions de perte (bornes hautes exclues).
pub const SEUIL_EXCELLENT: i32 = 20;
pub const SEUIL_BON: i32 = 50;
pub const SEUIL_IMPRECISION: i32 = 150;
pub const SEUIL_ERREUR: i32 = 300;

// ---------------------------------------------------------------------------
// Scores
// ---------------------------------------------------------------------------

/// Score UCI brut → centipions, du point de vue du CAMP AU TRAIT (le signe
/// n'est pas touché ici). `Mat(0)` = déjà maté → −MAT_CP.
pub fn cp_du_score(s: ScoreUci) -> i32 {
    match s {
        ScoreUci::Cp(x) => x.clamp(-MAT_CP, MAT_CP),
        // n > 0 : le trait mate en n → MAT_CP − n.
        // n <= 0 : le trait est maté en |n| → −(MAT_CP − |n|) = −(MAT_CP + n).
        ScoreUci::Mat(n) if n > 0 => MAT_CP - n.min(MAT_CP),
        ScoreUci::Mat(n) => -(MAT_CP + n.max(-MAT_CP)),
    }
}

/// Centipions du point de vue du trait → point de vue des BLANCS.
pub fn vers_blancs(cp_trait: i32, trait_blanc: bool) -> i32 {
    if trait_blanc {
        cp_trait
    } else {
        -cp_trait
    }
}

/// Perte d'un coup, en centipions, du point de vue du camp qui vient de jouer.
///
/// `avant` = évaluation de la position AVANT le coup, `apres` = évaluation de
/// la position APRÈS, toutes deux CÔTÉ BLANCS. `trait_blanc` dit qui a joué.
/// Les deux évaluations sont d'abord plafonnées (voir `PLAFOND_PERTE`) ; le
/// résultat est borné à 0 : gagner de l'évaluation n'est pas une perte
/// négative, c'est simplement une perte nulle (le moteur peut « voir mieux »
/// d'une analyse à l'autre — profondeur, fenêtre, hasard du temps —, et ce
/// bruit ne doit pas créer de crédit compensant de vraies erreurs).
pub fn perte_cp(avant: i32, apres: i32, trait_blanc: bool) -> i32 {
    let a = vers_blancs(avant.clamp(-PLAFOND_PERTE, PLAFOND_PERTE), trait_blanc);
    let b = vers_blancs(apres.clamp(-PLAFOND_PERTE, PLAFOND_PERTE), trait_blanc);
    (a - b).max(0)
}

// ---------------------------------------------------------------------------
// Classement d'un coup
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classement {
    Meilleur,
    Excellent,
    Bon,
    Imprecision,
    Erreur,
    Gaffe,
}

impl Classement {
    pub fn nom(self) -> &'static str {
        match self {
            Classement::Meilleur => "meilleur",
            Classement::Excellent => "excellent",
            Classement::Bon => "bon",
            Classement::Imprecision => "imprecision",
            Classement::Erreur => "erreur",
            Classement::Gaffe => "gaffe",
        }
    }

    pub fn depuis_nom(s: &str) -> Option<Classement> {
        Some(match s {
            "meilleur" => Classement::Meilleur,
            "excellent" => Classement::Excellent,
            "bon" => Classement::Bon,
            "imprecision" => Classement::Imprecision,
            "erreur" => Classement::Erreur,
            "gaffe" => Classement::Gaffe,
            _ => return None,
        })
    }
}

/// Classement d'un coup : `identique` (le coup joué EST le meilleur coup du
/// moteur) l'emporte sur la perte — deux coups peuvent valoir exactement
/// autant, et une perte de 3 cp sur le coup que le moteur lui-même recommande
/// n'est que le bruit d'une analyse à l'autre, pas une imprécision.
pub fn classement(perte: i32, identique: bool) -> Classement {
    if identique {
        Classement::Meilleur
    } else if perte < SEUIL_EXCELLENT {
        Classement::Excellent
    } else if perte < SEUIL_BON {
        Classement::Bon
    } else if perte < SEUIL_IMPRECISION {
        Classement::Imprecision
    } else if perte < SEUIL_ERREUR {
        Classement::Erreur
    } else {
        Classement::Gaffe
    }
}

// ---------------------------------------------------------------------------
// Phase de la partie
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Ouverture,
    Milieu,
    Transition,
    Finale,
}

impl Phase {
    pub fn nom(self) -> &'static str {
        match self {
            Phase::Ouverture => "ouverture",
            Phase::Milieu => "milieu",
            Phase::Transition => "transition",
            Phase::Finale => "finale",
        }
    }

    pub fn depuis_nom(s: &str) -> Option<Phase> {
        Some(match s {
            "ouverture" => Phase::Ouverture,
            "milieu" => Phase::Milieu,
            "transition" => Phase::Transition,
            "finale" => Phase::Finale,
            _ => return None,
        })
    }

    /// Les quatre phases dans l'ordre de la partie (rendu du tableau de bord).
    pub const TOUTES: [Phase; 4] = [
        Phase::Ouverture,
        Phase::Milieu,
        Phase::Transition,
        Phase::Finale,
    ];
}

/// Phase de la position d'où part le coup numéro `ply` (1-based, donc `ply−1`
/// demi-coups déjà joués), avec `pieces` pièces sur l'échiquier (rois compris).
///
/// Bornes du contrat, ORDRE DE TEST explicite car les critères se recouvrent :
/// 1. ouverture : `ply <= 12` (les 6 premiers coups) OU `pieces >= 28` — le
///    second critère rattrape les ouvertures lentes où l'on n'a rien échangé au
///    13e demi-coup, et c'est bien la même chose qu'on juge (théorie, mise en
///    place), pas encore un milieu de partie ;
/// 2. finale : `pieces <= 9` ;
/// 3. transition : `pieces <= 14` (soit 10..14) ;
/// 4. milieu : le reste (`pieces >= 15`).
/// Les trois derniers critères partitionnent 1..32 sans trou ni recouvrement,
/// donc toute position reçoit exactement une phase. Bornes du contrat gardées
/// telles quelles : elles séparent déjà proprement les régimes où l'on veut
/// savoir si le champion pèche (finale = tables Syzygy, milieu = recherche).
pub fn phase(ply: u32, pieces: u32) -> Phase {
    if ply <= 12 || pieces >= 28 {
        Phase::Ouverture
    } else if pieces <= 9 {
        Phase::Finale
    } else if pieces <= 14 {
        Phase::Transition
    } else {
        Phase::Milieu
    }
}

/// Camp qui joue le demi-coup `ply` (1-based) : les BLANCS jouent les plis
/// impairs. Rendu « champion » / « fantome » selon la couleur du champion.
pub fn camp(ply: u32, champion_blanc: bool) -> &'static str {
    let trait_blanc = ply % 2 == 1;
    if trait_blanc == champion_blanc {
        "champion"
    } else {
        "fantome"
    }
}

// ---------------------------------------------------------------------------
// Annotation d'un pli + persistance CSV
// ---------------------------------------------------------------------------

/// Entête de models/arbitre.csv (une ligne par pli annoté).
pub const ENTETE_CSV: &str = "partie,ply,camp,phase,eval_avant_cp,meilleur,joue,perte_cp,classement";

/// Annotation complète d'un demi-coup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Annotation {
    pub partie: u32,
    /// Numéro du demi-coup, 1-based (le pli 1 est le premier coup des blancs).
    pub ply: u32,
    /// "champion" ou "fantome".
    pub camp: String,
    pub phase: Phase,
    /// Évaluation de la position AVANT le coup, CÔTÉ BLANCS, non plafonnée.
    pub eval_avant: i32,
    /// Évaluation de la position APRÈS le coup, CÔTÉ BLANCS, non plafonnée.
    /// Absente du CSV (déductible du pli suivant) : gardée pour le JSON.
    pub eval_apres: i32,
    /// Meilleur coup du moteur, en SAN. Vide si le moteur n'en a pas rendu.
    pub meilleur: String,
    /// Coup réellement joué, en SAN.
    pub joue: String,
    pub perte_cp: i32,
    pub classement: Classement,
}

impl Annotation {
    /// Ligne CSV (sans saut de ligne). Aucun échappement nécessaire : le SAN
    /// ne contient jamais de virgule, et tous les autres champs sont des
    /// entiers ou des mots-clés du crate.
    pub fn ligne_csv(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{}",
            self.partie,
            self.ply,
            self.camp,
            self.phase.nom(),
            self.eval_avant,
            self.meilleur,
            self.joue,
            self.perte_cp,
            self.classement.nom()
        )
    }

    /// Relit une ligne CSV.
    ///
    /// `eval_apres` n'est PAS consignée (le contrat fixe les neuf colonnes) :
    /// elle est ici REPLIÉE sur `eval_avant ∓ perte_cp`, ce qui est faux dès
    /// que le camp au trait a GAGNÉ de l'évaluation — `perte_cp` est bornée à
    /// 0 (voir `perte_cp`), donc la soustraction rend `eval_avant` inchangée
    /// alors que la vraie valeur est ailleurs (mesuré sur un CSV réel : un
    /// tiers des plis, jusqu'à 19 cp d'écart). Ce repli ne sert donc que pour
    /// le DERNIER pli d'une partie : pour tous les autres, `lit_csv` recolle
    /// ensuite la valeur EXACTE, qui est l'`eval_avant` du pli suivant
    /// (`recolle_eval_apres`). Champ d'affichage uniquement : `perte_cp` et
    /// tout le résumé viennent du CSV et restent justes dans tous les cas.
    pub fn depuis_csv(ligne: &str) -> Option<Annotation> {
        let c: Vec<&str> = ligne.trim_end().split(',').collect();
        if c.len() < 9 {
            return None;
        }
        let eval_avant = c[4].trim().parse::<i32>().ok()?;
        let perte_cp = c[7].trim().parse::<i32>().ok()?;
        let ply = c[1].trim().parse::<u32>().ok()?;
        // Repli : eval_apres ≈ eval_avant − perte du point de vue du joueur.
        // Faux quand la perte a été bornée à 0 ou plafonnée — corrigé juste
        // après par `recolle_eval_apres` partout où le pli suivant est connu.
        let trait_blanc = ply % 2 == 1;
        let eval_apres = if trait_blanc {
            eval_avant - perte_cp
        } else {
            eval_avant + perte_cp
        };
        Some(Annotation {
            partie: c[0].trim().parse().ok()?,
            ply,
            camp: c[2].trim().to_string(),
            phase: Phase::depuis_nom(c[3].trim())?,
            eval_avant,
            eval_apres,
            meilleur: c[5].trim().to_string(),
            joue: c[6].trim().to_string(),
            perte_cp,
            classement: Classement::depuis_nom(c[8].trim())?,
        })
    }

    pub fn json(&self) -> serde_json::Value {
        json!({
            "ply": self.ply,
            "camp": self.camp,
            "phase": self.phase.nom(),
            "eval_avant": self.eval_avant,
            "eval_apres": self.eval_apres,
            "meilleur": self.meilleur,
            "joue": self.joue,
            "perte_cp": self.perte_cp,
            "classement": self.classement.nom(),
        })
    }
}

/// Relit models/arbitre.csv en entier (entête et lignes illisibles ignorées) :
/// c'est la REPRISE — l'arbitre ne réanalysera aucun pli déjà consigné, et
/// reconstruit le tableau de bord de la partie en cours sans un seul appel au
/// moteur. Une ligne tronquée (arrêt brutal en pleine écriture) est simplement
/// sautée : elle sera réanalysée (`prochain_pli` rend le premier TROU).
///
/// Deux recollages après lecture :
/// - DÉDOUBLONNAGE sur (partie, ply), la dernière ligne l'emportant. Sans lui,
///   deux arbitres lancés par mégarde sur le même CSV (ou une reprise après un
///   arrêt brutal en pleine écriture puis réécriture) feraient compter le même
///   pli deux fois dans `resume` : perte moyenne et compteurs faux.
/// - `recolle_eval_apres` : l'évaluation d'après-coup exacte, prise sur le pli
///   suivant (voir `Annotation::depuis_csv`).
pub fn lit_csv(contenu: &str) -> Vec<Annotation> {
    let mut lues: Vec<Annotation> = contenu
        .lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with("partie,"))
        .filter_map(Annotation::depuis_csv)
        .collect();
    // Dédoublonnage en gardant la DERNIÈRE occurrence (l'analyse la plus
    // récente) et l'ordre du fichier pour les survivantes.
    let mut vues: std::collections::HashSet<(u32, u32)> = std::collections::HashSet::new();
    let mut garde: Vec<bool> = vec![false; lues.len()];
    for (i, a) in lues.iter().enumerate().rev() {
        garde[i] = vues.insert((a.partie, a.ply));
    }
    let mut a_garder = garde.into_iter();
    lues.retain(|_| a_garder.next().unwrap_or(true));
    recolle_eval_apres(&mut lues);
    lues
}

/// Remplace l'`eval_apres` repliée de chaque annotation par sa valeur EXACTE
/// quand elle est connue : l'évaluation de la position après le pli `n` EST
/// celle d'avant le pli `n+1` de la même partie (c'est déjà l'économie
/// d'analyses du binaire, ici appliquée à la relecture du CSV). Les plis sans
/// successeur consigné — le dernier de chaque partie — gardent leur repli.
pub fn recolle_eval_apres(annotations: &mut [Annotation]) {
    let avant: std::collections::HashMap<(u32, u32), i32> = annotations
        .iter()
        .map(|a| ((a.partie, a.ply), a.eval_avant))
        .collect();
    for a in annotations.iter_mut() {
        if let Some(&e) = avant.get(&(a.partie, a.ply + 1)) {
            a.eval_apres = e;
        }
    }
}

/// Premier demi-coup NON encore annoté de la partie `partie` : le premier TROU
/// (et 1 si la partie est absente du CSV), en ignorant en plus les plis listés
/// dans `ignores`.
///
/// Le premier trou, et non `max(ply) + 1` : une ligne déchirée par un arrêt
/// brutal est sautée par `lit_csv`, et si elle n'était pas la dernière, un
/// `max + 1` laisserait ce pli manquant pour toujours (résumé incomplet, case
/// vide dans le panneau). Robuste au désordre comme aux doublons.
///
/// `ignores` sert aux plis ABANDONNÉS par le binaire (position que le moteur
/// refuse obstinément d'évaluer) : sans cette liste, le premier trou serait
/// éternellement le même et la boucle n'avancerait plus.
pub fn prochain_pli_hors(annotations: &[Annotation], partie: u32, ignores: &[u32]) -> u32 {
    let mut vus: std::collections::HashSet<u32> = annotations
        .iter()
        .filter(|a| a.partie == partie)
        .map(|a| a.ply)
        .collect();
    vus.extend(ignores.iter().copied());
    let mut pli = 1;
    while vus.contains(&pli) {
        pli += 1;
    }
    pli
}

/// `prochain_pli_hors` sans pli abandonné (cas courant).
pub fn prochain_pli(annotations: &[Annotation], partie: u32) -> u32 {
    prochain_pli_hors(annotations, partie, &[])
}

/// Le CSV décrit-il bien LE match diffusé ? `san` est l'historique diffusé et
/// `plis` le nombre de demi-coups exploitables (SAN et FEN disponibles). Deux
/// signes que le CSV est périmé (un nouveau match reparti à la partie 1, par
/// exemple) :
/// - un pli consigné dont le coup joué diffère de `san` ;
/// - un pli consigné AU-DELÀ de l'historique diffusé (on n'annote jamais un
///   pli qui n'a pas été joué : c'est donc une partie plus longue d'un autre
///   match) ;
/// - un `ply` nul, qui ne peut venir que d'une ligne aberrante (1-based).
///
/// `annotations` doit être restreint à la partie diffusée par l'appelant.
/// Vecteur vide → cohérent (rien à contredire).
pub fn csv_coherent(annotations: &[Annotation], san: &[String], plis: usize) -> bool {
    annotations.iter().all(|a| {
        let index = (a.ply as usize).checked_sub(1); // ply 1-based ; 0 = aberrant
        index.is_some_and(|i| i < plis && san.get(i) == Some(&a.joue))
    })
}

// ---------------------------------------------------------------------------
// Rattrapage de fin de partie : relecture du PGN
// ---------------------------------------------------------------------------

/// Coups SAN du movetext d'un PGN (celui qu'écrit `match.rs::ecrit_pgn`, et
/// les variantes usuelles) : les entêtes `[...]`, les numéros de coup (`12.`,
/// `12...`, `12.e4` collé) et le résultat final sont écartés.
///
/// Sert au RATTRAPAGE de fin de partie : match.rs écrit le PGN juste avant de
/// publier le premier état de la partie suivante, donc au moment où l'arbitre
/// voit le numéro de partie changer, le PGN complet de la partie close est déjà
/// sur le disque — c'est la seule source qui contienne à coup sûr ses derniers
/// plis (models/match_live.json, lui, a déjà été écrasé).
///
/// Aucun support des commentaires `{...}`, variantes `(...)` ni NAG `$n` : le
/// PGN visé n'en contient pas, et un jeton inattendu sera simplement rejeté
/// plus loin par `fens_de_coups`, qui s'arrête au premier coup injouable.
pub fn coups_du_pgn(contenu: &str) -> Vec<String> {
    let mut coups = Vec::new();
    for ligne in contenu.lines() {
        let l = ligne.trim();
        if l.is_empty() || l.starts_with('[') {
            continue;
        }
        for jeton in l.split_whitespace() {
            // « 12. », « 12... » ou « 12.e4 » : on ne garde que ce qui suit le
            // dernier point.
            let mot = match jeton.rsplit('.').next() {
                Some(m) => m,
                None => continue,
            };
            // Un SAN commence toujours par une lettre (pièce, colonne ou O-O) :
            // écarte « 1-0 », « 1/2-1/2 », « * », « $3 » et les numéros seuls.
            if mot.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                coups.push(mot.to_string());
            }
        }
    }
    coups
}

/// FEN de la position AVANT chaque coup, position initiale incluse — même
/// convention et même format que `history_fen` de models/match_live.json
/// (`EnPassantMode::Legal`), pour que le cache d'analyses du binaire serve
/// indifféremment les deux sources.
///
/// Rend `coups.len() + 1` FEN quand tous les coups sont jouables, et s'arrête
/// au premier coup illisible ou illégal (PGN d'un autre format) : l'appelant
/// compare les longueurs et retombe alors sur l'instantané du direct.
pub fn fens_de_coups(coups: &[String]) -> Vec<String> {
    let mut pos = Chess::default();
    let mut fens = vec![Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string()];
    for c in coups {
        let Ok(san) = San::from_ascii(c.as_bytes()) else { break };
        let Ok(coup) = san.to_move(&pos) else { break };
        let Ok(suivante) = pos.play(&coup) else { break };
        pos = suivante;
        fens.push(Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string());
    }
    fens
}

/// Statistiques d'un camp sur un ensemble d'annotations.
fn stats_camp(annotations: &[Annotation], camp: &str) -> serde_json::Value {
    let plis: Vec<&Annotation> = annotations.iter().filter(|a| a.camp == camp).collect();
    let n = plis.len();
    let somme: i64 = plis.iter().map(|a| a.perte_cp as i64).sum();
    let compte = |c: Classement| plis.iter().filter(|a| a.classement == c).count();
    json!({
        "plis": n,
        "perte_moyenne": moyenne(somme, n),
        "gaffes": compte(Classement::Gaffe),
        "erreurs": compte(Classement::Erreur),
        "imprecisions": compte(Classement::Imprecision),
        "meilleurs": compte(Classement::Meilleur),
    })
}

/// Moyenne arrondie au dixième de centipion, ou `null` sur un ensemble vide :
/// un 0 afficherait « perte moyenne nulle » là où l'on n'a rien mesuré.
fn moyenne(somme: i64, n: usize) -> serde_json::Value {
    if n == 0 {
        serde_json::Value::Null
    } else {
        json!((somme as f64 / n as f64 * 10.0).round() / 10.0)
    }
}

/// Résumé publié dans models/match_arbitre.json : par camp, et par phase pour
/// le CHAMPION (c'est la question de Martin : OÙ pèche-t-il ?).
/// `plis_par_phase_champion` accompagne les moyennes : 12 cp de perte sur
/// 3 plis de finale et sur 40 plis de milieu ne se lisent pas pareil.
pub fn resume(annotations: &[Annotation]) -> serde_json::Value {
    let mut par_phase = serde_json::Map::new();
    let mut plis_par_phase = serde_json::Map::new();
    for ph in Phase::TOUTES {
        let plis: Vec<&Annotation> = annotations
            .iter()
            .filter(|a| a.camp == "champion" && a.phase == ph)
            .collect();
        let somme: i64 = plis.iter().map(|a| a.perte_cp as i64).sum();
        par_phase.insert(ph.nom().to_string(), moyenne(somme, plis.len()));
        plis_par_phase.insert(ph.nom().to_string(), json!(plis.len()));
    }
    json!({
        "par_camp": {
            "champion": stats_camp(annotations, "champion"),
            "fantome": stats_camp(annotations, "fantome"),
        },
        "par_phase_champion": par_phase,
        "plis_par_phase_champion": plis_par_phase,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Conversion des scores UCI (cp et mat) vers les centipions puis vers le
    /// point de vue des BLANCS.
    #[test]
    fn conversion_scores_uci() {
        // Centipions : transmis tels quels côté trait.
        assert_eq!(cp_du_score(ScoreUci::Cp(35)), 35);
        assert_eq!(cp_du_score(ScoreUci::Cp(-120)), -120);
        // Mats : ±(32000 − n), l'ordre reste total (mat en 1 > mat en 7).
        assert_eq!(cp_du_score(ScoreUci::Mat(1)), 31_999);
        assert_eq!(cp_du_score(ScoreUci::Mat(7)), 31_993);
        assert!(cp_du_score(ScoreUci::Mat(1)) > cp_du_score(ScoreUci::Mat(7)));
        assert_eq!(cp_du_score(ScoreUci::Mat(-3)), -31_997);
        assert!(cp_du_score(ScoreUci::Mat(-1)) < cp_du_score(ScoreUci::Mat(-7)));
        // Mat 0 : déjà maté, le pire score possible.
        assert_eq!(cp_du_score(ScoreUci::Mat(0)), -MAT_CP);

        // Point de vue : trait aux blancs → inchangé ; aux noirs → renversé.
        assert_eq!(vers_blancs(35, true), 35);
        assert_eq!(vers_blancs(35, false), -35);
        assert_eq!(vers_blancs(-31_997, false), 31_997);
        // Le trait NOIR annonce « je mate en 2 » → −31998 côté blancs.
        assert_eq!(vers_blancs(cp_du_score(ScoreUci::Mat(2)), false), -31_998);
    }

    /// perte_cp : point de vue du joueur, jamais négative, plafonnée.
    #[test]
    fn calcul_perte_cp() {
        // Blancs : +50 → +10 = 40 cp perdus.
        assert_eq!(perte_cp(50, 10, true), 40);
        // Blancs qui GAGNENT de l'éval : perte nulle, pas de crédit négatif.
        assert_eq!(perte_cp(10, 90, true), 0);
        // Noirs : l'éval côté blancs qui MONTE est une perte pour eux.
        assert_eq!(perte_cp(-50, 120, false), 170);
        assert_eq!(perte_cp(-50, -200, false), 0);
        // Égalité parfaite : 0 des deux côtés.
        assert_eq!(perte_cp(0, 0, true), 0);
        assert_eq!(perte_cp(0, 0, false), 0);
        // Plafond : « +300 puis maté » ne compte pas 32 300 cp, mais l'écart
        // entre +300 et le plafond bas — bien au-delà du seuil « gaffe ».
        assert_eq!(perte_cp(300, -31_996, true), 300 + PLAFOND_PERTE);
        // Deux positions déjà hors plafond : écart nul, pas de bruit.
        assert_eq!(perte_cp(31_990, 31_996, true), 0);
        assert_eq!(perte_cp(-31_990, -31_996, false), 0);
    }

    /// Classement : identité au meilleur coup d'abord, puis les seuils.
    #[test]
    fn classement_par_seuils() {
        // Le coup du moteur reste « meilleur » quelle que soit la perte
        // mesurée (bruit d'analyse d'une profondeur à l'autre).
        assert_eq!(classement(0, true), Classement::Meilleur);
        assert_eq!(classement(400, true), Classement::Meilleur);
        // Bornes hautes EXCLUES : 19 excellent, 20 bon, 49 bon, 50 imprécision…
        assert_eq!(classement(0, false), Classement::Excellent);
        assert_eq!(classement(19, false), Classement::Excellent);
        assert_eq!(classement(20, false), Classement::Bon);
        assert_eq!(classement(49, false), Classement::Bon);
        assert_eq!(classement(50, false), Classement::Imprecision);
        assert_eq!(classement(149, false), Classement::Imprecision);
        assert_eq!(classement(150, false), Classement::Erreur);
        assert_eq!(classement(299, false), Classement::Erreur);
        assert_eq!(classement(300, false), Classement::Gaffe);
        assert_eq!(classement(5_000, false), Classement::Gaffe);
        // Aller-retour des noms (CSV → mémoire).
        for c in [
            Classement::Meilleur,
            Classement::Excellent,
            Classement::Bon,
            Classement::Imprecision,
            Classement::Erreur,
            Classement::Gaffe,
        ] {
            assert_eq!(Classement::depuis_nom(c.nom()), Some(c));
        }
        assert_eq!(Classement::depuis_nom("brillant"), None);
    }

    /// Phase : l'ouverture l'emporte, puis les bornes de matériel partitionnent.
    #[test]
    fn classement_de_phase() {
        // Ouverture par le PLI (12 demi-coups), même matériel réduit.
        assert_eq!(phase(1, 32), Phase::Ouverture);
        assert_eq!(phase(12, 20), Phase::Ouverture);
        // Ouverture par le MATÉRIEL (28+ pièces), même passé le 12e pli.
        assert_eq!(phase(13, 28), Phase::Ouverture);
        assert_eq!(phase(40, 30), Phase::Ouverture);
        // Milieu : 15..27 pièces après le 12e pli.
        assert_eq!(phase(13, 27), Phase::Milieu);
        assert_eq!(phase(30, 15), Phase::Milieu);
        // Transition : 10..14.
        assert_eq!(phase(30, 14), Phase::Transition);
        assert_eq!(phase(30, 10), Phase::Transition);
        // Finale : ≤ 9.
        assert_eq!(phase(30, 9), Phase::Finale);
        assert_eq!(phase(200, 3), Phase::Finale);
        // Partition totale : toute position reçoit une phase et une seule.
        for ply in 1..=60u32 {
            for pieces in 2..=32u32 {
                let p = phase(ply, pieces);
                assert!(Phase::TOUTES.contains(&p), "pli {ply}, {pieces} pièces");
            }
        }
    }

    /// Camp d'un demi-coup : les blancs jouent les plis impairs.
    #[test]
    fn camp_du_pli() {
        assert_eq!(camp(1, true), "champion");
        assert_eq!(camp(2, true), "fantome");
        assert_eq!(camp(1, false), "fantome");
        assert_eq!(camp(2, false), "champion");
        assert_eq!(camp(29, true), "champion");
        assert_eq!(camp(30, true), "fantome");
    }

    fn annot(partie: u32, ply: u32, champion_blanc: bool, perte: i32, ph: Phase) -> Annotation {
        Annotation {
            partie,
            ply,
            camp: camp(ply, champion_blanc).to_string(),
            phase: ph,
            eval_avant: 25,
            eval_apres: 25 - perte,
            meilleur: "Nf3".into(),
            joue: "Nc3".into(),
            perte_cp: perte,
            classement: classement(perte, false),
        }
    }

    /// Aller-retour CSV puis REPRISE depuis un CSV PARTIEL : l'arbitre repart
    /// au premier pli non consigné, sans réanalyser les précédents, et sait
    /// qu'une partie absente du fichier commence au pli 1.
    #[test]
    fn reprise_idempotente_depuis_csv_partiel() {
        let a = annot(1, 7, true, 42, Phase::Ouverture);
        let relu = Annotation::depuis_csv(&a.ligne_csv()).expect("ligne relisible");
        // Tous les champs consignés reviennent à l'identique.
        assert_eq!(relu.partie, a.partie);
        assert_eq!(relu.ply, a.ply);
        assert_eq!(relu.camp, a.camp);
        assert_eq!(relu.phase, a.phase);
        assert_eq!(relu.eval_avant, a.eval_avant);
        assert_eq!(relu.meilleur, a.meilleur);
        assert_eq!(relu.joue, a.joue);
        assert_eq!(relu.perte_cp, a.perte_cp);
        assert_eq!(relu.classement, a.classement);
        // eval_apres reconstruite : ply 7 = trait aux blancs.
        assert_eq!(relu.eval_apres, a.eval_apres);

        // CSV partiel : entête + 3 plis de la partie 1, une ligne TRONQUÉE
        // (arrêt brutal en pleine écriture) et une ligne vide.
        let mut csv = String::from(ENTETE_CSV);
        for ply in 1..=3u32 {
            csv.push('\n');
            csv.push_str(&annot(1, ply, true, 10 * ply as i32, Phase::Ouverture).ligne_csv());
        }
        csv.push_str("\n1,4,champion,ouverture,25,Nf3\n\n");
        let lues = lit_csv(&csv);
        assert_eq!(lues.len(), 3, "l'entête, la ligne tronquée et la vide sautées");
        // Reprise : on repart au pli 4, et la partie 2 (absente) au pli 1.
        assert_eq!(prochain_pli(&lues, 1), 4);
        assert_eq!(prochain_pli(&lues, 2), 1);
        assert_eq!(prochain_pli(&[], 1), 1);
        // Désordre et doublons (reprise après arrêt brutal) : le PREMIER trou.
        let desordre = vec![
            annot(1, 3, true, 0, Phase::Ouverture),
            annot(1, 1, true, 0, Phase::Ouverture),
            annot(1, 3, true, 0, Phase::Ouverture),
        ];
        assert_eq!(prochain_pli(&desordre, 1), 2, "le pli 2 manque : on le rebouche");
        // Une fois le trou comblé, on repart bien après le dernier pli.
        let mut complet = desordre.clone();
        complet.push(annot(1, 2, true, 0, Phase::Ouverture));
        assert_eq!(prochain_pli(&complet, 1), 4);
        // Pli ABANDONNÉ (moteur muet sur cette position) : ignoré comme s'il
        // était annoté, sinon la boucle resterait bloquée dessus à jamais.
        assert_eq!(prochain_pli_hors(&desordre, 1, &[2]), 4);
        assert_eq!(prochain_pli_hors(&[], 1, &[1, 2]), 3);
    }

    /// Trou au MILIEU du CSV (ligne déchirée par un arrêt brutal, puis d'autres
    /// plis écrits par-dessus) : il est réanalysé, pas abandonné.
    #[test]
    fn trou_au_milieu_rebouche() {
        let mut csv = String::from(ENTETE_CSV);
        for ply in [1u32, 2, 4, 5] {
            csv.push('\n');
            csv.push_str(&annot(1, ply, true, 10, Phase::Ouverture).ligne_csv());
        }
        let lues = lit_csv(&csv);
        assert_eq!(lues.len(), 4);
        assert_eq!(prochain_pli(&lues, 1), 3, "le pli 3 manquant est repris");
    }

    /// Lignes en double sur le même (partie, ply) — deux arbitres lancés par
    /// mégarde sur le même CSV : un seul exemplaire survit, sinon le résumé
    /// compterait ces plis deux fois.
    #[test]
    fn doublons_ecartes_a_la_relecture() {
        let mut a = annot(1, 1, true, 100, Phase::Ouverture);
        let mut csv = String::from(ENTETE_CSV);
        csv.push('\n');
        csv.push_str(&a.ligne_csv());
        csv.push('\n');
        a.perte_cp = 120; // seconde analyse du MÊME pli, valeur légèrement autre
        csv.push_str(&a.ligne_csv());
        csv.push('\n');
        csv.push_str(&annot(1, 2, true, 0, Phase::Ouverture).ligne_csv());
        let lues = lit_csv(&csv);
        assert_eq!(lues.len(), 2, "le pli 1 n'apparaît qu'une fois");
        assert_eq!(lues[0].ply, 1);
        assert_eq!(lues[0].perte_cp, 120, "la dernière analyse l'emporte");
        assert_eq!(lues[1].ply, 2);
        // Le résumé ne compte plus le pli 1 deux fois (pli 1 = champion,
        // pli 2 = fantôme, le champion ayant les blancs).
        let r = resume(&lues);
        assert_eq!(r["par_camp"]["champion"]["plis"], 1);
        assert_eq!(r["par_camp"]["champion"]["perte_moyenne"], 120.0);
        assert_eq!(r["par_camp"]["fantome"]["plis"], 1);
    }

    /// eval_apres : repliée par soustraction à la relecture, puis RECOLLÉE sur
    /// l'eval_avant du pli suivant, qui est la vraie valeur.
    #[test]
    fn eval_apres_recollee_depuis_le_pli_suivant() {
        // Pli 1 : les blancs GAGNENT de l'éval (perte bornée à 0) — le repli
        // rendrait 300 alors que la position d'après vaut 333.
        let mut a1 = annot(1, 1, true, 0, Phase::Ouverture);
        a1.eval_avant = 300;
        a1.eval_apres = 333;
        let mut a2 = annot(1, 2, true, 0, Phase::Ouverture);
        a2.eval_avant = 333;
        a2.eval_apres = 320;
        let csv = format!("{ENTETE_CSV}\n{}\n{}", a1.ligne_csv(), a2.ligne_csv());
        let lues = lit_csv(&csv);
        assert_eq!(lues[0].eval_avant, 300);
        assert_eq!(lues[0].eval_apres, 333, "valeur exacte, pas le repli à 300");
        // Dernier pli consigné : aucun successeur, le repli subsiste (ply 2 =
        // trait aux noirs → eval_avant + perte, ici perte nulle).
        assert_eq!(lues[1].eval_apres, 333);
        // perte_cp et le résumé, eux, viennent du CSV et sont inchangés.
        assert_eq!(lues[0].perte_cp, 0);
    }

    /// Cohérence CSV / match diffusé : c'est le seul verdict qui puisse
    /// déclencher l'archivage du CSV, il ne doit se déclarer périmé qu'à bon
    /// escient.
    #[test]
    fn coherence_csv_avec_le_direct() {
        let san: Vec<String> = ["e4", "c5", "Nf3"].iter().map(|s| s.to_string()).collect();
        let mut a = annot(1, 1, true, 0, Phase::Ouverture);
        a.joue = "e4".into();
        let mut b = annot(1, 2, true, 0, Phase::Ouverture);
        b.joue = "c5".into();

        // CSV vide : rien à contredire.
        assert!(csv_coherent(&[], &san, san.len()));
        // Plis consignés conformes à l'historique diffusé.
        assert!(csv_coherent(&[a.clone(), b.clone()], &san, san.len()));
        // Coup divergent au même pli : autre match.
        let mut divergent = b.clone();
        divergent.joue = "e5".into();
        assert!(!csv_coherent(&[a.clone(), divergent], &san, san.len()));
        // Pli AU-DELÀ de l'historique diffusé : autre match, plus long.
        let mut trop_loin = a.clone();
        trop_loin.ply = 4;
        assert!(!csv_coherent(&[trop_loin], &san, san.len()));
        // Pli consigné au-delà de ce que le direct expose VRAIMENT (SAN publié
        // mais FEN manquante : plis < san.len()).
        assert!(!csv_coherent(&[b.clone()], &san, 1));
        // ply 0 : ligne aberrante (les plis sont 1-based).
        let mut zero = a.clone();
        zero.ply = 0;
        assert!(!csv_coherent(&[zero], &san, san.len()));
    }

    /// Rattrapage de fin de partie : coups d'un PGN écrit par match.rs, puis
    /// FEN reconstruites — mêmes chaînes que history_fen du direct.
    #[test]
    fn relecture_du_pgn_et_reconstruction_des_fen() {
        let pgn = "[Event \"Match\"]\n[Result \"1-0\"]\n[Termination \"mat\"]\n\n\
                   1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 1-0\n";
        let coups = coups_du_pgn(pgn);
        assert_eq!(coups.len(), 10, "les 10 demi-coups, sans les numéros ni 1-0");
        assert_eq!(coups[0], "e4");
        assert_eq!(coups[5], "cxd4");
        assert_eq!(coups[9], "a6");
        // Numéros collés, roques, promotions et nulle : tolérés de la même façon.
        let colle = coups_du_pgn("1.e4 e5 2.O-O O-O-O 3.a8=Q+ Kb8 1/2-1/2\n");
        assert_eq!(colle, ["e4", "e5", "O-O", "O-O-O", "a8=Q+", "Kb8"]);
        // Les entêtes ne fournissent aucun coup, une partie vide non plus.
        assert!(coups_du_pgn("[White \"Champion e4\"]\n\n*\n").is_empty());

        // FEN : une de plus que de coups, la première étant la position
        // initiale, au format EXACT de models/match_live.json.
        let fens = fens_de_coups(&coups);
        assert_eq!(fens.len(), coups.len() + 1);
        assert_eq!(fens[0], "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(fens[1], "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1");
        // Coup injouable : on s'arrête là, sans paniquer.
        let boiteux = vec!["e4".to_string(), "e4".to_string()];
        assert_eq!(fens_de_coups(&boiteux).len(), 2);
        assert_eq!(fens_de_coups(&[]).len(), 1);
    }

    /// Résumé : moyennes par camp et par phase du champion, compteurs, et
    /// `null` (pas 0) là où rien n'a été mesuré.
    #[test]
    fn resume_par_camp_et_par_phase() {
        let annotations = vec![
            // Champion (blancs, plis impairs) : 0, 200 et 400 cp.
            annot(1, 1, true, 0, Phase::Ouverture),
            annot(1, 3, true, 200, Phase::Milieu),
            annot(1, 5, true, 400, Phase::Milieu),
            // Fantôme (plis pairs) : 10 et 30 cp.
            annot(1, 2, true, 10, Phase::Ouverture),
            annot(1, 4, true, 30, Phase::Milieu),
        ];
        let r = resume(&annotations);
        let champ = &r["par_camp"]["champion"];
        assert_eq!(champ["plis"], 3);
        assert_eq!(champ["perte_moyenne"], 200.0);
        assert_eq!(champ["gaffes"], 1); // 400 cp
        assert_eq!(champ["erreurs"], 1); // 200 cp
        assert_eq!(champ["imprecisions"], 0);
        assert_eq!(champ["meilleurs"], 0);
        let fant = &r["par_camp"]["fantome"];
        assert_eq!(fant["plis"], 2);
        assert_eq!(fant["perte_moyenne"], 20.0);
        // Par phase, CHAMPION seul : ouverture = 0, milieu = (200+400)/2.
        let ph = &r["par_phase_champion"];
        assert_eq!(ph["ouverture"], 0.0);
        assert_eq!(ph["milieu"], 300.0);
        assert!(ph["transition"].is_null(), "aucune transition mesurée");
        assert!(ph["finale"].is_null(), "aucune finale mesurée");
        assert_eq!(r["plis_par_phase_champion"]["milieu"], 2);
        assert_eq!(r["plis_par_phase_champion"]["finale"], 0);
        // Ensemble vide : tout à null / 0, aucune panique.
        let vide = resume(&[]);
        assert!(vide["par_camp"]["champion"]["perte_moyenne"].is_null());
        assert_eq!(vide["par_camp"]["fantome"]["plis"], 0);
    }
}
