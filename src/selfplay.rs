//! Partie d'auto-apprentissage : le réseau joue contre lui-même (1 pli + softmax
//! température pour explorer), chaque position visitée est étiquetée à la fin par
//! le résultat DU POINT DE VUE DU TRAIT de cette position (z ∈ {-1, 0, 1}).

use std::collections::HashMap;

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Position};

use crate::bots::{Bot, NetBot};
use crate::features::{encode, N_FEATURES};
use crate::nn::Mlp;

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
