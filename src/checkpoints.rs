//! Instantanés du modèle à des paliers de TEMPS D'ENTRAÎNEMENT CUMULÉ
//! (c'est la demande produit : pouvoir affronter « l'IA à 1 h », « à 3 h », ...).
//! L'état cumulé persiste dans models/state.json pour survivre aux reprises.

use serde::{Deserialize, Serialize};

/// Paliers en heures d'entraînement cumulées.
pub const MILESTONES_H: &[f64] = &[1.0, 3.0, 10.0, 30.0, 100.0];

#[derive(Serialize, Deserialize, Default, Clone)]
pub struct TrainState {
    pub trained_secs: f64,
    pub games: u64,
    pub positions: u64,
    pub cycles: u64,
}

impl TrainState {
    /// Charge `dir`/state.json, ou l'état vierge s'il n'existe pas.
    pub fn load(dir: &str) -> TrainState {
        let chemin = format!("{dir}/state.json");
        match std::fs::read_to_string(&chemin) {
            // Fichier illisible ou JSON invalide → on repart d'un état vierge.
            Ok(texte) => serde_json::from_str(&texte).unwrap_or_default(),
            Err(_) => TrainState::default(),
        }
    }

    pub fn save(&self, dir: &str) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let texte = serde_json::to_string_pretty(self)?;
        std::fs::write(format!("{dir}/state.json"), texte)
    }
}

/// Palier franchi entre deux instants (en heures), s'il y en a un.
/// C'est le plus PETIT palier contenu dans l'intervalle (before_h, after_h].
pub fn milestone_crossed(before_h: f64, after_h: f64) -> Option<f64> {
    // MILESTONES_H est trié croissant : le premier trouvé est le plus petit.
    MILESTONES_H
        .iter()
        .copied()
        .find(|&m| before_h < m && m <= after_h)
}

/// Chemin de l'instantané d'un palier : models/chess_t1h.bin, chess_t3h.bin,
/// chess_t10h.bin... (heures entières dans le nom).
pub fn milestone_path(dir: &str, hours: f64) -> String {
    format!("{dir}/chess_t{}h.bin", hours.round() as u64)
}

/// Chemin du dernier modèle : `dir`/chess_latest.bin.
pub fn latest_path(dir: &str) -> String {
    format!("{dir}/chess_latest.bin")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paliers_franchis() {
        assert_eq!(milestone_crossed(0.9, 1.1), Some(1.0));
        assert_eq!(milestone_crossed(1.1, 2.9), None);
        // Deux paliers dans l'intervalle (3 et 10) : on renvoie le plus petit.
        assert_eq!(milestone_crossed(2.9, 10.5), Some(3.0));
        assert_eq!(milestone_crossed(100.0, 200.0), None);
        // Borne exacte : le palier atteint pile est franchi.
        assert_eq!(milestone_crossed(0.5, 1.0), Some(1.0));
    }

    #[test]
    fn chemins() {
        assert_eq!(milestone_path("models", 1.0), "models/chess_t1h.bin");
        assert_eq!(milestone_path("models", 10.0), "models/chess_t10h.bin");
        assert_eq!(latest_path("models"), "models/chess_latest.bin");
    }

    #[test]
    fn etat_aller_retour() {
        let dir = std::env::temp_dir().join("echec_test_state");
        let dir = dir.to_string_lossy().to_string();
        let etat = TrainState { trained_secs: 12.5, games: 3, positions: 42, cycles: 7 };
        etat.save(&dir).expect("sauvegarde state.json");
        let relu = TrainState::load(&dir);
        assert_eq!(relu.trained_secs, 12.5);
        assert_eq!(relu.games, 3);
        assert_eq!(relu.positions, 42);
        assert_eq!(relu.cycles, 7);
        // Répertoire inexistant → état vierge.
        let vierge = TrainState::load("repertoire/inexistant");
        assert_eq!(vierge.games, 0);
    }
}
