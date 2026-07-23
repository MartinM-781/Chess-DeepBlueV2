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

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shakmaty::{Chess, Move};

use crate::arena;
use crate::bots::{Bot, MaterialBot, NetBot, RandomBot};
use crate::nn::Mlp;

/// Une ancre de l'échelle : nom, Elo estimé, profondeur du MaterialBot
/// (None = bot aléatoire).
pub struct Ancre {
    pub nom: &'static str,
    pub elo: f64,
    pub profondeur: Option<u32>,
}

/// Échelle maison, triée par force croissante.
pub const ANCRES: &[Ancre] = &[
    Ancre { nom: "aleatoire", elo: 400.0, profondeur: None },
    Ancre { nom: "materiel d1", elo: 800.0, profondeur: Some(1) },
    Ancre { nom: "materiel d2", elo: 1100.0, profondeur: Some(2) },
    Ancre { nom: "materiel d3", elo: 1350.0, profondeur: Some(3) },
    Ancre { nom: "materiel d4", elo: 1550.0, profondeur: Some(4) },
];

/// Score mesuré contre une ancre.
pub struct MesureAncre {
    pub nom: &'static str,
    pub elo_ancre: f64,
    /// Pourcentage de points dans [0, 1] (victoire 1, nulle 0.5).
    pub score: f64,
    pub parties: usize,
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

/// Joue `parties_par_ancre` parties contre chaque ancre (parallélisées par
/// arena::score) et renvoie les scores mesurés.
pub fn mesure(net: &Arc<Mlp>, depth: u32, parties_par_ancre: usize,
              graine: u64) -> Vec<MesureAncre> {
    ANCRES
        .iter()
        .enumerate()
        .map(|(k, a)| {
            let net_a = net.clone();
            let score = arena::score(
                move |g: u64| -> Box<dyn Bot> {
                    Box::new(BotReseau::new(net_a.clone(), g, depth))
                },
                |g: u64| -> Box<dyn Bot> {
                    match a.profondeur {
                        None => Box::new(RandomBot::new(g)),
                        Some(d) => Box::new(MaterialBot::new(g, d)),
                    }
                },
                parties_par_ancre,
                graine.wrapping_add(k as u64).wrapping_mul(0x9E37_79B9),
            ) as f64;
            MesureAncre { nom: a.nom, elo_ancre: a.elo, score, parties: parties_par_ancre }
        })
        .collect()
}

/// Ajuste l'Elo par maximum de vraisemblance binomiale sur toutes les mesures.
/// Les scores extrêmes sont adoucis (un 100 % sur n parties vaut « au plus
/// 1 - 1/(2n) ») pour garder la vraisemblance finie ; la log-vraisemblance est
/// unimodale en R → recherche ternaire sur [0, 3200].
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Scores synthétiques générés par la logistique d'un « vrai » Elo :
    /// l'ajustement doit le retrouver à quelques points près.
    #[test]
    fn retrouve_un_elo_connu() {
        for vrai in [600.0, 1000.0, 1400.0] {
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

    /// Tout écraser (100 % partout) doit donner une estimation au-dessus de la
    /// plus haute ancre, sans diverger.
    #[test]
    fn score_parfait_reste_borne() {
        let mesures: Vec<MesureAncre> = ANCRES
            .iter()
            .map(|a| MesureAncre { nom: a.nom, elo_ancre: a.elo, score: 1.0, parties: 24 })
            .collect();
        let estime = ajuste_elo(&mesures);
        assert!(estime > 1550.0 && estime <= 3200.0, "estimé {estime}");
    }
}
