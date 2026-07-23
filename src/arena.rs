//! Duels d'évaluation : score en pourcentage de points (victoire 1, nulle 0.5)
//! du camp A contre le camp B, couleurs alternées, parties jouées en parallèle
//! (rayon). Mêmes règles de nulle que selfplay (3 répétitions, 50 coups,
//! pat, matériel insuffisant, 300 plis max → nulle).

use std::collections::HashMap;

use rayon::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Position};

use crate::bots::Bot;

/// Plis maximum d'une partie d'arène avant arbitrage en nulle.
const MAX_PLIS_ARENE: u32 = 300;

fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Joue une partie blanc contre noir. Renvoie (résultat côté blancs, plis) :
/// 1.0 victoire blanche, -1.0 noire, 0.0 nulle. Règles de nulle : pat,
/// matériel insuffisant, 50 coups, 3e répétition, `max_plies` atteint.
fn jouer_duel<'a>(blanc: &'a mut dyn Bot, noir: &'a mut dyn Bot, max_plies: u32) -> (f32, u32) {
    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    let mut plies = 0u32;

    loop {
        let coups = pos.legal_moves();
        if coups.is_empty() {
            // Mat : le trait est perdant ; pat : nulle.
            let r = if pos.is_check() {
                if pos.turn() == Color::White { -1.0 } else { 1.0 }
            } else {
                0.0
            };
            return (r, plies);
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= max_plies {
            return (0.0, plies);
        }

        let bot = if pos.turn() == Color::White { &mut *blanc } else { &mut *noir };
        let m = bot.choose(&pos).expect("coups légaux non vides");
        pos = pos.play(&m).expect("coup légal");
        plies += 1;

        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            return (0.0, plies);
        }
    }
}

/// `a` et `b` fabriquent un bot frais pour une graine donnée (nécessaire pour
/// paralléliser sans partager d'état mutable). Partie i : A a les blancs si i pair.
/// Renvoie le pourcentage de points de A dans [0, 1].
pub fn score<A, B>(a: A, b: B, games: usize, base_seed: u64) -> f32
where
    A: Fn(u64) -> Box<dyn Bot> + Sync,
    B: Fn(u64) -> Box<dyn Bot> + Sync,
{
    if games == 0 {
        return 0.5;
    }
    // with_max_len(1) : UNE partie par tâche rayon. Sans ça, rayon regroupe les
    // parties en paquets séquentiels et les ouvriers qui ont fini n'ont plus
    // rien à voler — mesuré : 7 cœurs occupés sur 18 pendant les phases
    // parallèles, la durée d'une phase étant celle du paquet le plus lent.
    let total: f32 = (0..games)
        .into_par_iter()
        .with_max_len(1)
        .map(|i| {
            // Graines distinctes et déterministes pour chaque bot de chaque partie.
            let graine_a = base_seed.wrapping_add(2 * i as u64);
            let graine_b = base_seed.wrapping_add(2 * i as u64 + 1);
            let mut bot_a = a(graine_a);
            let mut bot_b = b(graine_b);
            // Couleurs alternées : A a les blancs si i pair.
            let resultat_a = if i % 2 == 0 {
                jouer_duel(bot_a.as_mut(), bot_b.as_mut(), MAX_PLIS_ARENE).0
            } else {
                -jouer_duel(bot_b.as_mut(), bot_a.as_mut(), MAX_PLIS_ARENE).0
            };
            // Points de A : victoire 1, nulle 0.5, défaite 0.
            (resultat_a + 1.0) / 2.0
        })
        .sum();
    total / games as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{MaterialBot, RandomBot};

    /// Avec les règles de nulle, une partie aléatoire se termine toujours
    /// naturellement (bien avant 600 plis).
    #[test]
    fn partie_aleatoire_se_termine() {
        for graine in 0..20u64 {
            let mut blanc = RandomBot::new(graine);
            let mut noir = RandomBot::new(graine + 1000);
            let (_r, plis) = jouer_duel(&mut blanc, &mut noir, 600);
            assert!(plis < 600, "partie non terminée naturellement (graine {graine}) : {plis} plis");
        }
    }

    /// MaterialBot profondeur 2 doit dominer largement le bot aléatoire.
    #[test]
    fn material_bat_random() {
        let s = score(
            |g| Box::new(MaterialBot::new(g, 2)) as Box<dyn Bot>,
            |g| Box::new(RandomBot::new(g)) as Box<dyn Bot>,
            20,
            42,
        );
        assert!(s >= 0.8, "score MaterialBot(d2) vs RandomBot trop faible : {s}");
    }
}
