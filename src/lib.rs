//! IA d'échecs par self-play : réseau de valeur (MLP maison) + moteur shakmaty.
//! Même philosophie que l'IA de poker (C:\dev\poker) : tout le chemin chaud compilé,
//! dashboard web (port 8778) avec plateau jouable et courbe d'entraînement,
//! adversaires à paliers de temps d'entraînement (1 h, 3 h, 10 h, ...).

pub mod arena;
pub mod bots;
pub mod checkpoints;
pub mod elo;
pub mod features;
pub mod nn;
pub mod nnue;
pub mod search;
pub mod selfplay;
pub mod uci;
