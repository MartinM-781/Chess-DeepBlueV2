//! Partie d'auto-apprentissage : le réseau joue contre lui-même (1 pli + softmax
//! température pour explorer), chaque position visitée est étiquetée à la fin par
//! le résultat DU POINT DE VUE DU TRAIT de cette position (z ∈ {-1, 0, 1}).
//! Variantes pilotées par la recherche : `play_training_game_recherche`
//! (cibles TD-leaf auto-étiquetées), `play_training_game_mentor` (coups
//! choisis par l'élève, étiquettes fournies par la recherche d'un mentor) et
//! `play_training_game_oracle` (étiquettes fournies par l'évaluation d'un
//! moteur UCI externe pleine force, voir uci.rs).

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;
use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Position};

use crate::bots::{echantillonne_scores_racine, Bot, NetBot};
use crate::direct;
use crate::features::{encode, N_FEATURES};
use crate::features_roi;
use crate::nn::{Mlp, SchemaFeatures};
use crate::search;
use crate::uci::UciEngine;

pub struct GameRecord {
    /// Positions encodées DENSES, concaténées (n_positions × N_FEATURES).
    /// Remplies UNIQUEMENT quand le réseau joueur est au schéma `Classique773`
    /// (vides pour `RoiZones8` : voir `actifs`).
    pub xs: Vec<f32>,
    /// Positions encodées CREUSES : une liste d'indices actifs
    /// (`features_roi::actifs`, perspective du trait) par position. Remplies
    /// UNIQUEMENT quand le réseau joueur est au schéma `RoiZones8` (vides
    /// pour `Classique773`). Entrée directe de `Mlp::train_batch_actifs`.
    pub actifs: Vec<Vec<u16>>,
    /// Étiquettes : résultat final vu du trait de chaque position (-1, 0, 1).
    pub zs: Vec<f32>,
    pub plies: u32,
    /// Résultat côté blancs : 1.0 victoire blanche, -1.0 noire, 0.0 nulle.
    pub result: f32,
}

/// Enregistre `pos` dans le format du schéma du réseau joueur : encodage dense
/// dans `xs` (via `buf`, redimensionné au besoin) pour `Classique773`, liste
/// d'indices actifs poussée dans `actifs` pour `RoiZones8`. Les deux formats
/// sont en perspective du TRAIT, comme les étiquettes zs.
fn enregistre_position(
    schema: SchemaFeatures,
    pos: &Chess,
    xs: &mut Vec<f32>,
    actifs: &mut Vec<Vec<u16>>,
    buf: &mut Vec<f32>,
) {
    match schema {
        SchemaFeatures::Classique773 => {
            buf.clear();
            buf.resize(N_FEATURES, 0.0);
            encode(pos, buf);
            xs.extend_from_slice(buf);
        }
        SchemaFeatures::RoiZones8 => {
            // ≤ 37 indices (32 pièces + 5 drapeaux).
            let mut a: Vec<u16> = Vec::with_capacity(40);
            features_roi::actifs(pos, &mut a);
            actifs.push(a);
        }
    }
}

/// Hachage zobrist 64 bits de la position (pour la détection de répétition).
fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Joue une partie complète. Règles de nulle à détecter :
/// pat, matériel insuffisant, règle des 50 coups (halfmoves >= 100),
/// 3e répétition (suivi des hachages zobrist de la partie), et
/// `max_plies` atteint (arbitrage en nulle).
pub fn play_training_game(net: &Mlp, seed: u64, temperature: f32,
                          max_plies: u32) -> GameRecord {
    let mut bot = NetBot::new(net, seed, temperature, 1);
    let mut pos = Chess::default();

    // Format d'enregistrement des positions : celui du schéma du réseau
    // (dense 773 dans xs, ou creux roi-zones dans actifs).
    let schema = net.schema();
    let mut xs: Vec<f32> = Vec::new();
    let mut actifs: Vec<Vec<u16>> = Vec::new();
    // Camp au trait de chaque position enregistrée (pour orienter z à la fin).
    let mut camps: Vec<Color> = Vec::new();
    let mut buf: Vec<f32> = Vec::new();

    // Compteur d'occurrences des positions de la partie (position initiale incluse).
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);

    let mut plies = 0u32;
    // Résultat côté blancs, fixé à la sortie de boucle.
    let result: f32;

    loop {
        let coups = pos.legal_moves();
        if coups.is_empty() {
            // Mat : le trait est perdant ; pat : nulle.
            result = if pos.is_check() {
                if pos.turn() == Color::White { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            break;
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= max_plies {
            result = 0.0;
            break;
        }

        // Enregistre la position AVANT le coup, du point de vue du trait.
        enregistre_position(schema, &pos, &mut xs, &mut actifs, &mut buf);
        camps.push(pos.turn());

        let m = bot.choose(&pos).expect("coups légaux non vides");
        pos = pos.play(&m).expect("coup légal");
        plies += 1;

        // 3e occurrence du même zobrist → nulle par répétition.
        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            result = 0.0;
            break;
        }
    }

    // z du point de vue du trait de CHAQUE position : si les blancs gagnent,
    // z = +1 pour les positions où les blancs étaient au trait, -1 sinon.
    let zs = camps
        .iter()
        .map(|c| if *c == Color::White { result } else { -result })
        .collect();

    GameRecord { xs, actifs, zs, plies, result }
}

/// Options du self-play piloté par la recherche (étage « Deep Blue »).
#[derive(Clone, Copy)]
pub struct OptionsRecherche {
    /// Budget de nœuds de recherche par coup.
    pub nodes_par_coup: u64,
    /// Température d'échantillonnage après l'ouverture (0 → meilleur coup).
    pub temperature: f32,
    /// Nombre de plis d'ouverture joués à `temperature_ouverture`.
    pub plis_ouverture: u32,
    /// Température (plus chaude) des plis d'ouverture, pour diversifier les débuts.
    pub temperature_ouverture: f32,
    /// Mélange TD-leaf : zs = lambda·z_final + (1-lambda)·v_racine.
    pub lambda: f32,
    /// Seuil d'arbitrage sur |v_racine| (score racine clampé à [-1,1]).
    pub seuil_arbitrage: f32,
    /// Nombre de plis CONSÉCUTIFS au-dessus du seuil pour arbitrer.
    pub plis_arbitrage: u32,
    /// Arbitrage en nulle au-delà de ce nombre de plis.
    pub max_plies: u32,
    /// Poids de l'étiqueteur EXTERNE (prof en mode mentoré, oracle UCI en
    /// mode oracle) dans la valeur mémorisée :
    /// v = poids_prof·v_étiqueteur + (1-poids_prof)·v_élève. 1.0 (défaut) =
    /// étiquettes externes pures (comportement historique). Desserrer vers
    /// 0.7 quand l'élève a convergé : sa propre recherche ré-entre dans les
    /// étiquettes, ce qui lui permet de DÉPASSER le prof au lieu d'en rester
    /// le clone. Sans effet hors modes mentoré et oracle (self-play
    /// classique : le chercheur s'étiquette lui-même).
    pub poids_prof: f32,
}

impl Default for OptionsRecherche {
    fn default() -> Self {
        OptionsRecherche {
            nodes_par_coup: 400,
            temperature: 0.2,
            plis_ouverture: 8,
            temperature_ouverture: 0.8,
            lambda: 0.3,
            seuil_arbitrage: 0.92,
            plis_arbitrage: 4,
            max_plies: 400,
            poids_prof: 1.0,
        }
    }
}

/// Table de recalibrage FIGÉE label → sortie du réseau (produite par
/// calibration.exe --fit) : g monotone croissante, appliquée par interpolation
/// linéaire UNIQUEMENT à l'étiquette oracle au moment de fabriquer la CIBLE
/// d'entraînement. Point de contrat : l'ARBITRAGE des parties et toute logique
/// de décision continuent d'utiliser l'étiquette BRUTE — g ne touche que les
/// cibles d'apprentissage. Vertu d'une table figée : l'échelle du réseau reste
/// stable pour toujours, le gradient ne transporte plus que de l'information
/// d'ordre.
pub struct Recalibrage {
    /// Nœuds (label, v), les DEUX colonnes strictement croissantes.
    noeuds: Vec<(f32, f32)>,
}

impl Recalibrage {
    /// Construit la table depuis des nœuds déjà en mémoire, en vérifiant la
    /// monotonie croissante STRICTE des deux colonnes (message clair sinon).
    pub fn depuis_noeuds(noeuds: Vec<(f32, f32)>) -> Result<Recalibrage, String> {
        if noeuds.len() < 2 {
            return Err(format!("au moins 2 noeuds attendus ({} lus)", noeuds.len()));
        }
        for k in 1..noeuds.len() {
            let (l0, v0) = noeuds[k - 1];
            let (l1, v1) = noeuds[k];
            if !(l1 > l0 && v1 > v0) {
                return Err(format!(
                    "monotonie croissante stricte violee au noeud {} : \
                     label {l0} -> {l1}, v {v0} -> {v1}",
                    k + 1
                ));
            }
            if !(l1.is_finite() && v1.is_finite() && l0.is_finite() && v0.is_finite()) {
                return Err(format!("valeur non finie au noeud {}", k + 1));
            }
        }
        Ok(Recalibrage { noeuds })
    }

    /// Charge un TSV « label<TAB>v » (une ligne par nœud, lignes vides et
    /// commentaires « # » ignorés), puis vérifie la monotonie stricte.
    pub fn charge(chemin: &str) -> Result<Recalibrage, String> {
        let texte = std::fs::read_to_string(chemin)
            .map_err(|e| format!("lecture impossible : {e}"))?;
        let mut noeuds: Vec<(f32, f32)> = Vec::new();
        for (i, ligne) in texte.lines().enumerate() {
            let l = ligne.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let mut champs = l.split('\t');
            let (a, b) = match (champs.next(), champs.next()) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return Err(format!(
                        "ligne {} : deux colonnes label<TAB>v attendues",
                        i + 1
                    ))
                }
            };
            let label: f32 = a.trim().parse().map_err(|_| {
                format!("ligne {} : label invalide « {a} »", i + 1)
            })?;
            let v: f32 = b.trim().parse().map_err(|_| {
                format!("ligne {} : v invalide « {b} »", i + 1)
            })?;
            noeuds.push((label, v));
        }
        Recalibrage::depuis_noeuds(noeuds)
    }

    /// g(label) par interpolation linéaire entre les nœuds (constante au-delà
    /// des bornes — la table couvre [-1, 1] en pratique).
    pub fn applique(&self, label: f32) -> f32 {
        let n = &self.noeuds;
        if label <= n[0].0 {
            return n[0].1;
        }
        if label >= n[n.len() - 1].0 {
            return n[n.len() - 1].1;
        }
        let i = n.partition_point(|&(x, _)| x <= label);
        let (x0, y0) = n[i - 1];
        let (x1, y1) = n[i];
        y0 + (y1 - y0) * (label - x0) / (x1 - x0)
    }

    /// Nombre de nœuds de la table.
    pub fn len(&self) -> usize {
        self.noeuds.len()
    }

    /// Premier et dernier nœud ((label, v) chacun), pour l'annonce d'en-tête.
    pub fn bornes(&self) -> ((f32, f32), (f32, f32)) {
        (self.noeuds[0], self.noeuds[self.noeuds.len() - 1])
    }
}

/// Cibles TD-leaf : zs[i] = lambda·z_final_i + (1-lambda)·v_racine_i, où
/// z_final_i est le résultat final vu du TRAIT de la position i (comme avant)
/// et v_racine_i le score racine de la recherche depuis cette position
/// (déjà du point de vue du trait, clampé à [-1,1]). Le gros du signal vient
/// de la recherche, le résultat final ancre la vérité terrain.
fn cibles_td_leaf(camps: &[Color], v_racines: &[f32], result: f32, lambda: f32) -> Vec<f32> {
    debug_assert_eq!(camps.len(), v_racines.len());
    camps
        .iter()
        .zip(v_racines)
        .map(|(c, v)| {
            let z_final = if *c == Color::White { result } else { -result };
            lambda * z_final + (1.0 - lambda) * v
        })
        .collect()
}

/// Partie d'auto-apprentissage pilotée par la RECHERCHE (TD-leaf).
///
/// - Ouverture (ply < plis_ouverture) : échantillonnage des scores racine à
///   `temperature_ouverture` ; ensuite à `temperature`.
/// - Chaque position enregistrée mémorise v_racine (score racine clampé [-1,1]).
/// - Arbitrage : |avantage blancs| >= seuil pendant `plis_arbitrage` plis
///   consécutifs → victoire du camp dominant (v_racine est du point de vue du
///   trait, qui alterne : converti côté blancs avant de compter).
/// - Règles de nulle habituelles inchangées (pat/matériel/50 coups/3 rép./max_plies).
/// - zs = cibles TD-leaf (voir `cibles_td_leaf`) ; `result` reste côté blancs.
pub fn play_training_game_recherche(
    recherche: &mut search::Recherche,
    seed: u64,
    opts: &OptionsRecherche,
) -> GameRecord {
    partie_recherche_interne(recherche, Etiqueteur::Aucun, Chess::default(),
                             opts.plis_ouverture, None, seed, opts, None)
}

/// Comme `play_training_game_recherche`, mais la partie démarre de
/// `depart.pos` (position d'ouverture du livre, finale générée, ...) et les
/// plis « chauds » (température d'ouverture) sont `depart.plis_chauds` au lieu
/// de `opts.plis_ouverture`. Tout le reste est identique — le hachage zobrist
/// de la position de départ est bien inséré dans le suivi des répétitions, et
/// `depart.etiquette` est retransmise au direct (clé « depart » de live.json).
pub fn play_training_game_recherche_depuis(
    recherche: &mut search::Recherche,
    depart: &crate::departs::Depart,
    seed: u64,
    opts: &OptionsRecherche,
) -> GameRecord {
    partie_recherche_interne(recherche, Etiqueteur::Aucun, depart.pos.clone(),
                             depart.plis_chauds, Some(depart.etiquette), seed, opts, None)
}

/// Partie de self-play MENTORÉE — remède à la chambre d'écho du TD
/// auto-référentiel : l'ÉLÈVE choisit les coups (sa recherche, mêmes
/// températures/ouverture/échantillonnage que `play_training_game_recherche`),
/// mais la valeur mémorisée de CHAQUE position enregistrée est le score racine
/// de la recherche du PROF sur cette même position (mêmes limites de nœuds,
/// clampé [-1,1]) — v_prof[i] correspond à la position i AVANT le coup i,
/// aucun décalage — et l'arbitrage s'appuie lui aussi sur le v_racine du prof,
/// plus fiable. Les cibles restent `cibles_td_leaf(camps, v_prof, result,
/// lambda)`. Les choix de coups ne dépendent PAS du prof : avec le même réseau
/// des deux côtés, le déroulé est identique à `play_training_game_recherche`.
pub fn play_training_game_mentor(
    eleve: &mut search::Recherche,
    prof: &mut search::Recherche,
    seed: u64,
    opts: &OptionsRecherche,
) -> GameRecord {
    partie_recherche_interne(eleve, Etiqueteur::Prof(prof), Chess::default(),
                             opts.plis_ouverture, None, seed, opts, None)
}

/// Comme `play_training_game_mentor`, mais depuis `depart.pos` avec
/// `depart.plis_chauds` plis chauds (voir `play_training_game_recherche_depuis`).
pub fn play_training_game_mentor_depuis(
    eleve: &mut search::Recherche,
    prof: &mut search::Recherche,
    depart: &crate::departs::Depart,
    seed: u64,
    opts: &OptionsRecherche,
) -> GameRecord {
    partie_recherche_interne(eleve, Etiqueteur::Prof(prof), depart.pos.clone(),
                             depart.plis_chauds, Some(depart.etiquette), seed, opts, None)
}

/// Partie de self-play ORACLE — même déroulé que le mentorat (l'élève choisit
/// tous les coups avec sa recherche, mêmes températures/ouverture), mais la
/// valeur mémorisée de chaque position vient de l'ÉVALUATION d'un moteur UCI
/// externe pleine force (`UciEngine::evalue_fen` sur la FEN de la position
/// courante, AVANT le coup — aucun décalage). CONVENTION PARTAGÉE : le score
/// UCI est du point de vue du camp au trait, comme v_racine — aucun
/// renversement de signe. `poids_prof` mélange oracle/élève comme en
/// mentorat : v = poids_prof·v_oracle + (1-poids_prof)·v_élève, et
/// l'arbitrage s'appuie sur ce mélange. Si l'oracle ne répond pas (processus
/// mort ou FIGÉ au-delà de l'échéance de lecture — voir uci.rs —, ligne
/// imparsable) : repli silencieux sur le score de l'élève pour
/// CETTE position — la partie continue, jamais de panique.
/// `recalibrage` (Some = table g de calibration.exe --fit) ne transforme que
/// l'étiquette oracle ENTRANT DANS LES CIBLES d'entraînement — l'arbitrage
/// reste sur l'étiquette brute ; None = strictement aucun changement.
pub fn play_training_game_oracle(
    chercheur: &mut search::Recherche,
    oracle: &mut UciEngine,
    seed: u64,
    opts: &OptionsRecherche,
    recalibrage: Option<&Recalibrage>,
) -> GameRecord {
    partie_recherche_interne(chercheur, Etiqueteur::Oracle(oracle), Chess::default(),
                             opts.plis_ouverture, None, seed, opts, recalibrage)
}

/// Comme `play_training_game_oracle`, mais depuis `depart.pos` avec
/// `depart.plis_chauds` plis chauds (voir `play_training_game_recherche_depuis`).
pub fn play_training_game_oracle_depuis(
    chercheur: &mut search::Recherche,
    oracle: &mut UciEngine,
    depart: &crate::departs::Depart,
    seed: u64,
    opts: &OptionsRecherche,
    recalibrage: Option<&Recalibrage>,
) -> GameRecord {
    partie_recherche_interne(chercheur, Etiqueteur::Oracle(oracle), depart.pos.clone(),
                             depart.plis_chauds, Some(depart.etiquette), seed, opts, recalibrage)
}

/// « Qui étiquette » les positions du self-play piloté par la recherche : la
/// valeur mémorisée de chaque position (cibles TD-leaf) ET l'arbitrage
/// viennent de cette source. Interne — les fonctions publiques ci-dessus
/// choisissent la variante.
enum Etiqueteur<'a> {
    /// Le chercheur s'étiquette lui-même (self-play classique).
    Aucun,
    /// La recherche d'un réseau mentor étiquette (mêmes limites de nœuds).
    Prof(&'a mut search::Recherche),
    /// L'évaluation d'un moteur UCI externe pleine force étiquette
    /// (budget movetime fixé par `UciEngine::lance_pleine_force`).
    Oracle(&'a mut UciEngine),
}

/// Cœur commun du self-play piloté par la recherche, paramétré par « qui
/// étiquette » (voir `Etiqueteur`) : `chercheur` choisit les coups ;
/// l'étiqueteur fournit le v_racine mémorisé (cibles TD-leaf) ET l'arbitrage —
/// `Aucun` : le chercheur s'étiquette lui-même (self-play classique),
/// `Prof` : la recherche du prof étiquette les MÊMES positions avec les MÊMES
/// limites (mentorat), `Oracle` : l'évaluation d'un moteur UCI externe
/// étiquette (repli élève position par position si le moteur ne répond pas).
/// La partie démarre de `pos_depart` (position initiale pour les variantes
/// historiques, position du livre/finale pour les variantes `_depuis`) et
/// `plis_chauds` remplace `opts.plis_ouverture` comme durée de la phase à
/// `temperature_ouverture`. `etiquette_depart` (Some pour les variantes
/// `_depuis`, None sinon) est retransmise au direct sous la clé « depart »
/// de live.json — la page /live peut indiquer la provenance du départ.
fn partie_recherche_interne(
    chercheur: &mut search::Recherche,
    mut etiqueteur: Etiqueteur,
    pos_depart: Chess,
    plis_chauds: u32,
    etiquette_depart: Option<&str>,
    seed: u64,
    opts: &OptionsRecherche,
    recalibrage: Option<&Recalibrage>,
) -> GameRecord {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pos = pos_depart;
    // Une partie = une TT propre (killers et historique compris) — pour les
    // DEUX chercheurs en mode mentoré ; l'oracle reçoit l'équivalent UCI
    // (`ucinewgame`).
    chercheur.nouvelle_partie();
    match &mut etiqueteur {
        Etiqueteur::Prof(p) => p.nouvelle_partie(),
        // Un échec ici n'est PAS fatal : evalue_fen renverra None et le repli
        // élève s'appliquera position par position.
        Etiqueteur::Oracle(o) => {
            let _ = o.nouvelle_partie();
        }
        Etiqueteur::Aucun => {}
    }
    // Une source d'étiquettes EXTERNE existe-t-elle ? (jauge « prof » du direct)
    let a_etiqueteur = !matches!(etiqueteur, Etiqueteur::Aucun);

    let limites = search::Limites {
        max_noeuds: opts.nodes_par_coup,
        max_profondeur: 0,
        movetime_ms: 0,
    };

    // Direct : cette partie tente de « prendre le micro » — si obtenu, chaque
    // coup joué est publié dans live.json (page /live du dashboard). Pour
    // toutes les autres parties, le surcoût se limite à ce compare_exchange
    // raté (et à rien du tout si le direct n'est pas configuré).
    let journaliste = direct::prendre_le_micro();
    let cycle = direct::cycle_courant();
    // État du direct — rempli SEULEMENT micro en main (SAN et FEN coûtent).
    let mut historique_san: Vec<String> = Vec::new();
    let mut dernier_uci: Option<String> = None;
    // Dernières évaluations converties CÔTÉ BLANCS (élève, prof) : les jauges
    // de la page /live sont fixes, alors que le trait (et donc la perspective
    // des scores de recherche) alterne à chaque coup.
    let mut dernier_v_eleve: Option<f32> = None;
    let mut dernier_v_prof: Option<f32> = None;

    // Format d'enregistrement des positions : celui du schéma du réseau de
    // l'ÉLÈVE (celui qui sera entraîné sur ces positions) — dense 773 dans xs
    // ou creux roi-zones dans actifs. En mentorat, le prof peut être d'un
    // autre schéma : il ne fournit que les étiquettes, jamais les entrées.
    let schema = chercheur.net.schema();
    let mut xs: Vec<f32> = Vec::new();
    let mut actifs: Vec<Vec<u16>> = Vec::new();
    let mut camps: Vec<Color> = Vec::new();
    let mut v_racines: Vec<f32> = Vec::new();
    let mut buf: Vec<f32> = Vec::new();

    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);

    // Série de plis consécutifs où |avantage blancs| >= seuil, signée
    // (positif : les blancs dominent, négatif : les noirs dominent).
    let mut serie_arbitrage: i32 = 0;

    let mut plies = 0u32;
    let result: f32;

    loop {
        let coups = pos.legal_moves();
        if coups.is_empty() {
            // Mat : le trait est perdant ; pat : nulle.
            result = if pos.is_check() {
                if pos.turn() == Color::White { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            break;
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= opts.max_plies {
            result = 0.0;
            break;
        }

        let res = chercheur.cherche(&pos, limites);
        // v_racine mémorisé : valeur de l'ÉTIQUETEUR sur la position COURANTE
        // (aucun décalage : v_racines[i] ↔ position i AVANT le coup i), du
        // chercheur lui-même sans étiqueteur. v_cible = valeur qui entre dans
        // les CIBLES d'entraînement : identique à v_racine, sauf recalibrage
        // actif en branche oracle (g appliquée à l'étiquette seule).
        let (v_racine, v_cible) = match &mut etiqueteur {
            Etiqueteur::Aucun => {
                let v = res.score.clamp(-1.0, 1.0);
                (v, v)
            }
            // Mentorat : mélange prof/élève selon poids_prof (1.0 = prof pur,
            // comportement historique). Les deux scores sont du point de vue
            // du MÊME trait sur la MÊME position : le mélange est légitime.
            // Mêmes limites de nœuds que l'élève.
            Etiqueteur::Prof(p) => {
                let v_prof = p.cherche(&pos, limites).score.clamp(-1.0, 1.0);
                let v_eleve = res.score.clamp(-1.0, 1.0);
                let v = opts.poids_prof * v_prof + (1.0 - opts.poids_prof) * v_eleve;
                (v, v)
            }
            // Oracle : évaluation du moteur externe sur la MÊME position (FEN
            // mode Legal, comme partout). Convention UCI = score du point de
            // vue du camp au trait, identique à v_racine : AUCUN renversement
            // de signe. evalue_fen renvoie déjà une valeur dans [-1, 1]
            // (tanh(cp/300) ou ±1 sur les mats) ; poids_prof mélange
            // oracle/élève comme en mentorat. Moteur muet/mort → repli sur le
            // score de l'élève pour CETTE position, la partie CONTINUE.
            // Recalibrage : g ne transforme l'étiquette oracle QUE dans la
            // valeur destinée aux cibles — l'arbitrage (et le direct) restent
            // sur l'étiquette brute, c'est un point de contrat.
            Etiqueteur::Oracle(o) => {
                let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
                let v_eleve = res.score.clamp(-1.0, 1.0);
                match o.evalue_fen(&fen) {
                    Some(v_oracle) => {
                        let brut = opts.poids_prof * v_oracle + (1.0 - opts.poids_prof) * v_eleve;
                        let cible = match recalibrage {
                            Some(g) => {
                                opts.poids_prof * g.applique(v_oracle)
                                    + (1.0 - opts.poids_prof) * v_eleve
                            }
                            None => brut,
                        };
                        (brut, cible)
                    }
                    None => (v_eleve, v_eleve),
                }
            }
        };

        // Direct : évaluations mémorisées AVANT tout break (l'arbitrage peut
        // terminer la partie plus bas), converties du point de vue du trait
        // vers le point de vue des BLANCS. v_eleve = recherche de l'élève
        // (celui qui choisit les coups), v_prof = étiqueteur externe (mentor
        // ou oracle ; null en self-play classique).
        if journaliste.is_some() {
            let signe = if pos.turn() == Color::White { 1.0 } else { -1.0 };
            dernier_v_eleve = Some(res.score.clamp(-1.0, 1.0) * signe);
            dernier_v_prof = a_etiqueteur.then(|| v_racine * signe);
        }

        // Enregistre la position AVANT le coup, du point de vue du trait.
        enregistre_position(schema, &pos, &mut xs, &mut actifs, &mut buf);
        camps.push(pos.turn());
        // C'est v_cible (recalibrée le cas échéant) qui entre dans les zs.
        v_racines.push(v_cible);

        // Arbitrage : v_racine est du point de vue du trait, qui ALTERNE —
        // converti en « avantage blancs » avant de compter les plis consécutifs.
        let v_blancs = if pos.turn() == Color::White { v_racine } else { -v_racine };
        if v_blancs >= opts.seuil_arbitrage {
            serie_arbitrage = if serie_arbitrage >= 0 { serie_arbitrage + 1 } else { 1 };
        } else if v_blancs <= -opts.seuil_arbitrage {
            serie_arbitrage = if serie_arbitrage <= 0 { serie_arbitrage - 1 } else { -1 };
        } else {
            serie_arbitrage = 0;
        }
        if opts.plis_arbitrage > 0 && serie_arbitrage.unsigned_abs() >= opts.plis_arbitrage {
            result = if serie_arbitrage > 0 { 1.0 } else { -1.0 };
            break;
        }

        // Ouverture diversifiée (plis « chauds »), puis régime normal.
        let t = if plies < plis_chauds {
            opts.temperature_ouverture
        } else {
            opts.temperature
        };
        let m = if t > 0.0 {
            echantillonne_scores_racine(&res.scores_racine, t, &mut rng)
                .or(res.coup)
                .expect("coups légaux non vides")
        } else {
            res.coup.expect("coups légaux non vides")
        };
        // Direct : phase du coup joué (avant l'incrément de plies) et SAN
        // calculé AVANT de jouer (il a besoin de la position de départ).
        let phase = if plies < plis_chauds { "ouverture" } else { "normale" };
        if journaliste.is_some() {
            historique_san.push(San::from_move(&pos, &m).to_string());
        }
        pos = pos.play(&m).expect("coup légal");
        plies += 1;

        // Direct : publie la position APRÈS le coup, SAN cumulés, v_* du
        // point de vue des blancs (mémorisés au moment de la recherche),
        // étiquette du départ (null pour les variantes historiques).
        if let Some(j) = &journaliste {
            dernier_uci = Some(m.to_uci(CastlingMode::Standard).to_string());
            let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
            j.publie_avec_depart(cycle, plies, &fen, dernier_uci.as_deref(),
                                 &historique_san, dernier_v_eleve, dernier_v_prof,
                                 phase, None, etiquette_depart);
        }

        // 3e occurrence du même zobrist → nulle par répétition.
        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            result = 0.0;
            break;
        }
    }

    // Direct : publication finale avec le résultat — la page /live garde la
    // position finale à l'écran le temps qu'une autre partie prenne le micro
    // (rendu au Drop du Journaliste, à la sortie de cette fonction).
    if let Some(j) = &journaliste {
        let resultat = if result > 0.0 {
            "1-0"
        } else if result < 0.0 {
            "0-1"
        } else {
            "1/2-1/2"
        };
        let phase = if plies < plis_chauds { "ouverture" } else { "normale" };
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        j.publie_avec_depart(cycle, plies, &fen, dernier_uci.as_deref(),
                             &historique_san, dernier_v_eleve, dernier_v_prof,
                             phase, Some(resultat), etiquette_depart);
    }

    let zs = cibles_td_leaf(&camps, &v_racines, result, opts.lambda);
    GameRecord { xs, actifs, zs, plies, result }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::Mlp;
    use std::sync::Arc;

    /// Mélange lambda vérifié sur un cas construit à la main.
    #[test]
    fn cibles_td_leaf_melange_lambda() {
        // Blancs gagnent (result = 1.0). Position 0 : trait blanc, v = 0.5 ;
        // position 1 : trait noir, v = -0.4.
        let camps = [Color::White, Color::Black];
        let v_racines = [0.5f32, -0.4f32];
        let zs = cibles_td_leaf(&camps, &v_racines, 1.0, 0.3);
        // zs[0] = 0.3*(+1) + 0.7*0.5  = 0.65
        // zs[1] = 0.3*(-1) + 0.7*(-0.4) = -0.58
        assert!((zs[0] - 0.65).abs() < 1e-6, "zs[0] = {}", zs[0]);
        assert!((zs[1] + 0.58).abs() < 1e-6, "zs[1] = {}", zs[1]);
        // lambda = 1 → résultat pur ; lambda = 0 → recherche pure.
        let zs1 = cibles_td_leaf(&camps, &v_racines, 1.0, 1.0);
        assert_eq!(zs1, vec![1.0, -1.0]);
        let zs0 = cibles_td_leaf(&camps, &v_racines, 1.0, 0.0);
        assert!((zs0[0] - 0.5).abs() < 1e-6 && (zs0[1] + 0.4).abs() < 1e-6);
    }

    /// Une partie recherche à 400 nœuds/coup se termine (< 400 plis).
    #[test]
    fn partie_recherche_se_termine() {
        let net = Arc::new(Mlp::new(42));
        let mut recherche = search::Recherche::new(net, 16);
        let opts = OptionsRecherche::default();
        let rec = play_training_game_recherche(&mut recherche, 7, &opts);
        assert!(rec.plies < 400, "partie trop longue : {} plis", rec.plies);
        assert_eq!(rec.xs.len(), rec.zs.len() * N_FEATURES);
        assert!(rec.result == 1.0 || rec.result == 0.0 || rec.result == -1.0);
        // Cibles bornées : |z| <= lambda + (1-lambda) = 1.
        assert!(rec.zs.iter().all(|z| z.abs() <= 1.0 + 1e-6));
    }

    /// BOUT EN BOUT JOUET, schéma ROI-ZONES, régime 1-PLI : un réseau
    /// `RoiZones8` neuf joue une partie complète de self-play (NetBot →
    /// `nn::evalue_position` → chemin creux), les positions sont enregistrées
    /// en indices actifs (xs dense VIDE), les zs sont finis, et un pas de
    /// `train_batch_actifs` sur ces données produit une loss finie.
    #[test]
    fn partie_1pli_roi_zones_bout_en_bout() {
        use crate::features_roi::N_FEATURES_ROI;
        // Petit réseau (couche cachée de 32) : le test reste rapide.
        let mut net = Mlp::new_roi_zones(&[N_FEATURES_ROI, 32, 1], 42);
        let rec = play_training_game(&net, 7, 0.5, 120);
        assert!(rec.plies > 0 && rec.plies <= 120, "plies aberrant : {}", rec.plies);
        // Schéma creux : les indices actifs remplacent l'encodage dense.
        assert!(rec.xs.is_empty(), "xs dense doit rester vide en RoiZones8");
        assert_eq!(rec.actifs.len(), rec.zs.len());
        assert!(rec.zs.iter().all(|z| z.is_finite() && z.abs() <= 1.0 + 1e-6));
        for a in &rec.actifs {
            // Au moins les deux rois, au plus 32 pièces + 5 drapeaux.
            assert!(a.len() >= 2 && a.len() <= 37, "position à {} indices", a.len());
            assert!(a.iter().all(|&i| usize::from(i) < N_FEATURES_ROI));
        }
        // Un pas d'entraînement creux sur la partie entière : loss finie.
        let lots: Vec<(Vec<u16>, f32)> = rec
            .actifs
            .iter()
            .cloned()
            .zip(rec.zs.iter().copied())
            .collect();
        let loss = net.train_batch_actifs(&lots, 1e-3);
        assert!(loss.is_finite(), "loss non finie : {loss}");
    }

    /// BOUT EN BOUT JOUET, schéma ROI-ZONES, régime RECHERCHE (le test du
    /// contrat) : réseau `RoiZones8` neuf → une partie de self-play recherche
    /// à 200 nœuds/coup se termine et produit des zs finis.
    ///
    /// IGNORÉ tant que le dernier chantier amont n'est pas livré :
    /// src/search.rs (GELÉ, autre escouade) : la parité debug de
    /// `evaluer()` recalcule en `encode` 773 + `forward_one`, ce qui
    /// paniquerait en mode test avec un réseau 6149. (src/nnue.rs est
    /// livré : `EvalIncrementale` accepte les deux schémas, parité roi8
    /// couverte par la batterie 7a-7f de nnue.rs.)
    /// Dès que search.rs sera dégelé et sa parité debug routée par schéma,
    /// retirer l'#[ignore] — le test est prêt.
    #[test]
    fn partie_recherche_roi_zones_se_termine() {
        use crate::features_roi::N_FEATURES_ROI;
        let net = Arc::new(Mlp::new_roi_zones(&[N_FEATURES_ROI, 32, 1], 42));
        let mut recherche = search::Recherche::new(net, 16);
        let mut opts = OptionsRecherche::default();
        opts.nodes_par_coup = 200;
        let rec = play_training_game_recherche(&mut recherche, 7, &opts);
        assert!(rec.plies > 0 && rec.plies <= opts.max_plies,
                "nombre de plis aberrant : {}", rec.plies);
        assert!(rec.xs.is_empty(), "xs dense doit rester vide en RoiZones8");
        assert_eq!(rec.actifs.len(), rec.zs.len());
        assert!(rec.zs.iter().all(|z| z.is_finite() && z.abs() <= 1.0 + 1e-6));
    }

    /// Recalibrage : monotonie stricte exigée, interpolation linéaire exacte
    /// entre les nœuds, prolongement constant hors bornes.
    #[test]
    fn recalibrage_interpolation_et_monotonie() {
        // Table valide : g(-1)=-1, g(0)=-0.1, g(1)=0.9.
        let g = Recalibrage::depuis_noeuds(vec![(-1.0, -1.0), (0.0, -0.1), (1.0, 0.9)])
            .expect("table valide");
        assert_eq!(g.len(), 3);
        assert!((g.applique(0.0) + 0.1).abs() < 1e-6);
        // Milieu du premier segment : (-1 + -0.1)/2 = -0.55.
        assert!((g.applique(-0.5) + 0.55).abs() < 1e-6);
        // Milieu du second : (-0.1 + 0.9)/2 = 0.4.
        assert!((g.applique(0.5) - 0.4).abs() < 1e-6);
        // Hors bornes : prolongement constant.
        assert_eq!(g.applique(-2.0), -1.0);
        assert_eq!(g.applique(2.0), 0.9);
        // Monotonie violée (v égaux) : refusée avec un message clair.
        let err = Recalibrage::depuis_noeuds(vec![(-1.0, 0.0), (0.0, 0.0)]);
        assert!(err.is_err());
        // Moins de 2 nœuds : refusé.
        assert!(Recalibrage::depuis_noeuds(vec![(0.0, 0.0)]).is_err());
    }

    /// L'arbitrage raccourcit significativement les parties à fort
    /// déséquilibre : longueur moyenne avec/sans arbitrage sur 5 parties,
    /// graine fixe (seuil abaissé pour déclencher avec un réseau non entraîné).
    #[test]
    fn arbitrage_raccourcit_les_parties() {
        let net = Arc::new(Mlp::new(42));
        let mut recherche = search::Recherche::new(net, 16);
        let mut opts_sans = OptionsRecherche::default();
        opts_sans.plis_arbitrage = 0; // arbitrage désactivé
        let mut opts_avec = OptionsRecherche::default();
        opts_avec.seuil_arbitrage = 0.5;
        opts_avec.plis_arbitrage = 2;

        let total = |recherche: &mut search::Recherche, opts: &OptionsRecherche| -> u32 {
            (0..5u64).map(|g| play_training_game_recherche(recherche, 100 + g, opts).plies).sum()
        };
        let sans = total(&mut recherche, &opts_sans);
        let avec = total(&mut recherche, &opts_avec);
        assert!(
            avec < sans,
            "l'arbitrage devrait raccourcir : avec = {} plis, sans = {} plis",
            avec,
            sans
        );
    }

    /// Une partie mentorée (réseaux élève et prof DISTINCTS) à 300 nœuds/coup
    /// se termine et produit un enregistrement cohérent.
    #[test]
    fn partie_mentor_se_termine() {
        let mut eleve = search::Recherche::new(Arc::new(Mlp::new(42)), 16);
        let mut prof = search::Recherche::new(Arc::new(Mlp::new(43)), 16);
        let mut opts = OptionsRecherche::default();
        opts.nodes_par_coup = 300;
        let rec = play_training_game_mentor(&mut eleve, &mut prof, 7, &opts);
        assert!(rec.plies > 0 && rec.plies <= opts.max_plies,
                "nombre de plis aberrant : {}", rec.plies);
        assert_eq!(rec.xs.len(), rec.zs.len() * N_FEATURES);
        assert!(rec.result == 1.0 || rec.result == 0.0 || rec.result == -1.0);
        // Cibles bornées : |z| <= lambda + (1-lambda) = 1.
        assert!(rec.zs.iter().all(|z| z.abs() <= 1.0 + 1e-6));
    }

    /// Une partie oracle complète (élève 300 nœuds/coup, Stockfish pleine
    /// force movetime 10 ms) se termine et produit des cibles finies dans
    /// [-1, 1]. `cargo test --lib -- --ignored partie_oracle`.
    #[test]
    #[ignore = "nécessite engines/stockfish en local"]
    fn partie_oracle_se_termine() {
        let mut eleve = search::Recherche::new(Arc::new(Mlp::new(42)), 16);
        let mut oracle = UciEngine::lance_pleine_force(
            "engines/stockfish/stockfish-windows-x86-64-avx2.exe",
            10,
        )
        .expect("lancement de l'oracle");
        let mut opts = OptionsRecherche::default();
        opts.nodes_par_coup = 300;
        let rec = play_training_game_oracle(&mut eleve, &mut oracle, 7, &opts, None);
        assert!(rec.plies > 0 && rec.plies <= opts.max_plies,
                "nombre de plis aberrant : {}", rec.plies);
        assert_eq!(rec.xs.len(), rec.zs.len() * N_FEATURES);
        assert!(rec.result == 1.0 || rec.result == 0.0 || rec.result == -1.0);
        // Cibles finies et bornées : |z| <= lambda + (1-lambda) = 1.
        assert!(rec.zs.iter().all(|z| z.is_finite() && z.abs() <= 1.0 + 1e-6));
    }

    /// Une partie « depuis » une finale KRPvKR (variante recherche, 300
    /// nœuds/coup, SANS oracle réel) se termine proprement : mêmes invariants
    /// que le self-play classique, et le suivi des répétitions part bien de la
    /// position de départ. C'est le chemin qu'emprunte
    /// `play_training_game_oracle_depuis` (même cœur interne), testé ici sans
    /// dépendre d'un moteur UCI local.
    #[test]
    fn partie_depuis_finale_krpvkr_se_termine() {
        // Blancs : Rc1, Td1, Pc2 ; Noirs : Tc8, Rg8 — KRPvKR légal, trait blanc.
        let pos: Chess = "2r3k1/8/8/8/8/8/2P5/2KR4 w - - 0 1"
            .parse::<Fen>()
            .expect("FEN lisible")
            .into_position(CastlingMode::Standard)
            .expect("position légale");
        let depart = crate::departs::Depart {
            pos,
            etiquette: "finale:KRPvKR",
            plis_chauds: 0,
        };
        let net = Arc::new(Mlp::new(42));
        let mut recherche = search::Recherche::new(net, 16);
        let mut opts = OptionsRecherche::default();
        opts.nodes_par_coup = 300;
        let rec = play_training_game_recherche_depuis(&mut recherche, &depart, 7, &opts);
        assert!(rec.plies > 0 && rec.plies <= opts.max_plies,
                "nombre de plis aberrant : {}", rec.plies);
        assert_eq!(rec.xs.len(), rec.zs.len() * N_FEATURES);
        assert!(rec.result == 1.0 || rec.result == 0.0 || rec.result == -1.0);
        // Cibles bornées : |z| <= lambda + (1-lambda) = 1.
        assert!(rec.zs.iter().all(|z| z.abs() <= 1.0 + 1e-6));
    }

    /// Le mentor ne change QUE les étiquettes, jamais le déroulé : avec le
    /// MÊME réseau en élève et en prof, la partie mentorée reproduit
    /// exactement la partie auto-étiquetée (mêmes plies, même résultat, mêmes
    /// positions — et mêmes étiquettes : le prof, déterministe, recherche les
    /// mêmes positions avec la même TT que le chercheur solo), sur 3 graines.
    #[test]
    fn mentor_meme_deroule_que_recherche() {
        let net = Arc::new(Mlp::new(42));
        let mut opts = OptionsRecherche::default();
        opts.nodes_par_coup = 300;
        for g in [3u64, 5, 11] {
            let mut solo = search::Recherche::new(net.clone(), 16);
            let attendu = play_training_game_recherche(&mut solo, g, &opts);
            let mut eleve = search::Recherche::new(net.clone(), 16);
            let mut prof = search::Recherche::new(net.clone(), 16);
            let obtenu = play_training_game_mentor(&mut eleve, &mut prof, g, &opts);
            assert_eq!(attendu.plies, obtenu.plies, "graine {g} : plies");
            assert_eq!(attendu.result, obtenu.result, "graine {g} : result");
            assert_eq!(attendu.xs, obtenu.xs, "graine {g} : positions divergentes");
            assert_eq!(attendu.zs, obtenu.zs, "graine {g} : etiquettes divergentes");
        }
    }
}
