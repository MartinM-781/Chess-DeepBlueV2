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
/// bots maison (continuité historique), le haut (1700-2000) sur Stockfish
/// bridé — sans ces ancres hautes, le fit sature dès que le réseau écrase
/// tous les bots maison (~1600+) et la courbe perd toute résolution.
pub const ANCRES: &[Ancre] = &[
    Ancre { nom: "aleatoire", elo: 400.0, genre: GenreAncre::Maison { profondeur: None } },
    Ancre { nom: "materiel d1", elo: 800.0, genre: GenreAncre::Maison { profondeur: Some(1) } },
    Ancre { nom: "materiel d2", elo: 1100.0, genre: GenreAncre::Maison { profondeur: Some(2) } },
    Ancre { nom: "materiel d3", elo: 1350.0, genre: GenreAncre::Maison { profondeur: Some(3) } },
    Ancre { nom: "materiel d4", elo: 1550.0, genre: GenreAncre::Maison { profondeur: Some(4) } },
    Ancre { nom: "stockfish 1700", elo: 1700.0, genre: GenreAncre::Uci { elo_nominal: 1700, movetime_ms: 60 } },
    Ancre { nom: "stockfish 2000", elo: 2000.0, genre: GenreAncre::Uci { elo_nominal: 2000, movetime_ms: 60 } },
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

/// Joue `parties_par_ancre` parties contre chaque ancre MAISON (parallélisées
/// par arena::score) et renvoie les scores mesurés. Les ancres UCI de la liste
/// sont ignorées ici (elles exigent un chemin moteur : voir mesure_uci) — les
/// ancres maison venant en tête de liste, les indices de dérivation de graine
/// (et donc les duels) sont bit-à-bit identiques à l'historique.
pub fn mesure(net: &Arc<Mlp>, depth: u32, parties_par_ancre: usize,
              graine: u64) -> Vec<MesureAncre> {
    ANCRES
        .iter()
        .filter_map(|a| match a.genre {
            GenreAncre::Maison { profondeur } => Some((a, profondeur)),
            GenreAncre::Uci { .. } => None,
        })
        .enumerate()
        .map(|(k, (a, profondeur))| {
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
                parties_par_ancre,
                graine.wrapping_add(k as u64).wrapping_mul(0x9E37_79B9),
            ) as f64;
            MesureAncre { nom: a.nom, elo_ancre: a.elo, score, parties: parties_par_ancre }
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

/// Cœur de `mesure_uci`, liste d'ancres paramétrable (testabilité : les tests
/// jouent une ancre courte au lieu des 2 × 24 parties de production).
fn mesure_uci_liste<F>(fabrique: F, chemin_moteur: &str, ancres: &[Ancre],
                       parties_par_ancre: usize, graine: u64) -> Vec<MesureAncre>
where
    F: Fn(u64) -> Box<dyn Bot> + Sync,
{
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
    ancres
        .iter()
        .filter_map(|a| match a.genre {
            GenreAncre::Uci { elo_nominal, movetime_ms } => Some((a, elo_nominal, movetime_ms)),
            GenreAncre::Maison { .. } => None,
        })
        .enumerate()
        .filter_map(|(k, (a, elo_nominal, movetime_ms))| {
            // Clamp aux bornes annoncées (Stockfish : 1320..3190) ; l'Elo
            // EFFECTIVEMENT appliqué alimente le fit, pas le nominal.
            let elo_reel = elo_nominal.clamp(elo_min, elo_max);
            if elo_reel != elo_nominal {
                println!("  echelle Elo : {} clampe a UCI_Elo {elo_reel}", a.nom);
            }
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
                    parties_par_ancre,
                    graine.wrapping_add(0x5F00 + k as u64).wrapping_mul(0x9E37_79B9),
                ) as f64
            }));
            match resultat {
                Ok(score) => {
                    println!(
                        "  echelle Elo : {} -> {:.0} % ({} parties)",
                        a.nom, score * 100.0, parties_par_ancre
                    );
                    std::io::stdout().flush().ok();
                    Some(MesureAncre {
                        nom: a.nom,
                        elo_ancre: elo_reel as f64,
                        score,
                        parties: parties_par_ancre,
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
