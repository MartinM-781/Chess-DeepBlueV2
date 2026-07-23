//! Partie d'auto-apprentissage : le réseau joue contre lui-même (1 pli + softmax
//! température pour explorer), chaque position visitée est étiquetée à la fin par
//! le résultat DU POINT DE VUE DU TRAIT de cette position (z ∈ {-1, 0, 1}).

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Position};

use crate::bots::{echantillonne_scores_racine, Bot, NetBot};
use crate::features::{encode, N_FEATURES};
use crate::nn::Mlp;
use crate::search;

pub struct GameRecord {
    /// Positions encodées, concaténées (n_positions × N_FEATURES).
    pub xs: Vec<f32>,
    /// Étiquettes : résultat final vu du trait de chaque position (-1, 0, 1).
    pub zs: Vec<f32>,
    pub plies: u32,
    /// Résultat côté blancs : 1.0 victoire blanche, -1.0 noire, 0.0 nulle.
    pub result: f32,
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

    let mut xs: Vec<f32> = Vec::new();
    // Camp au trait de chaque position enregistrée (pour orienter z à la fin).
    let mut camps: Vec<Color> = Vec::new();
    let mut buf = vec![0.0f32; N_FEATURES];

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
        encode(&pos, &mut buf);
        xs.extend_from_slice(&buf);
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

    GameRecord { xs, zs, plies, result }
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
        }
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
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pos = Chess::default();
    // Une partie = une TT propre (killers et historique compris).
    recherche.nouvelle_partie();

    let limites = search::Limites {
        max_noeuds: opts.nodes_par_coup,
        max_profondeur: 0,
        movetime_ms: 0,
    };

    let mut xs: Vec<f32> = Vec::new();
    let mut camps: Vec<Color> = Vec::new();
    let mut v_racines: Vec<f32> = Vec::new();
    let mut buf = vec![0.0f32; N_FEATURES];

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

        let res = recherche.cherche(&pos, limites);
        let v_racine = res.score.clamp(-1.0, 1.0);

        // Enregistre la position AVANT le coup, du point de vue du trait.
        encode(&pos, &mut buf);
        xs.extend_from_slice(&buf);
        camps.push(pos.turn());
        v_racines.push(v_racine);

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

        // Ouverture diversifiée, puis régime normal.
        let t = if plies < opts.plis_ouverture {
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

    let zs = cibles_td_leaf(&camps, &v_racines, result, opts.lambda);
    GameRecord { xs, zs, plies, result }
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
}
