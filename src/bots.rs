//! Adversaires : bot aléatoire, bot matériel (alpha-bêta), bot réseau.
//! Tous renvoient None uniquement s'il n'existe aucun coup légal.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use shakmaty::{Chess, Move, Position};

use crate::features::{encode, N_FEATURES};
use crate::nn::Mlp;

pub trait Bot {
    fn choose(&mut self, pos: &Chess) -> Option<Move>;
}

/// Coup légal uniformément aléatoire.
pub struct RandomBot {
    pub rng: StdRng,
}

impl RandomBot {
    pub fn new(seed: u64) -> Self {
        RandomBot { rng: StdRng::seed_from_u64(seed) }
    }
}

impl Bot for RandomBot {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        pos.legal_moves().choose(&mut self.rng).cloned()
    }
}

/// Différence de matériel seule (nous - eux), du point de vue du trait, en pions.
fn materiel_trait(pos: &Chess) -> f32 {
    // Valeur d'un camp : P=1, N=3, B=3.15, R=5, Q=9 (le roi ne compte pas).
    let valeur = |c: shakmaty::Color| {
        let m = pos.board().material_side(c);
        m.pawn as f32
            + m.knight as f32 * 3.0
            + m.bishop as f32 * 3.15
            + m.rook as f32 * 5.0
            + m.queen as f32 * 9.0
    };
    let nous = pos.turn();
    valeur(nous) - valeur(!nous)
}

/// Évaluation matérielle simple, du point de vue du TRAIT, en pions :
/// P=1, N=3, B=3.15, R=5, Q=9 (+ petit bonus de mobilité). Utilisée par
/// MaterialBot et comme terme d'appoint du NetBot.
pub fn material_eval(pos: &Chess) -> f32 {
    // Mobilité : 0.01 × nombre de coups légaux du trait (une seule génération).
    materiel_trait(pos) + 0.01 * pos.legal_moves().len() as f32
}

/// Tolérance d'égalité pour le départage aléatoire des coups à la racine.
const EPS_EGALITE: f32 = 1e-6;

/// Négamax alpha-bêta sur l'évaluation matérielle. Mat détecté exactement :
/// score -(1000 - ply) pour le camp maté, si bien que les mats courts sont
/// préférés. Pat, matériel insuffisant et règle des 50 coups → 0.
fn negamax_materiel(pos: &Chess, depth: u32, ply: i32, mut alpha: f32, beta: f32) -> f32 {
    // Mat/pat testés AVANT la règle des 50 coups : un mat délivré pile au
    // 100e demi-coup est un mat pour les boucles de jeu (selfplay, arena),
    // la recherche doit rendre le même verdict.
    let mut coups = pos.legal_moves();
    if coups.is_empty() {
        // Mat (le trait est perdant) ou pat (nulle).
        return if pos.is_check() { -(1000.0 - ply as f32) } else { 0.0 };
    }
    if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
        return 0.0;
    }
    if depth == 0 {
        // On réutilise la liste déjà générée pour la mobilité (pas de double
        // génération de coups, material_eval en referait une).
        return materiel_trait(pos) + 0.01 * coups.len() as f32;
    }
    // Prises d'abord : améliore nettement l'élagage sans changer la valeur.
    coups.sort_unstable_by_key(|m| !m.is_capture());
    let mut best = f32::NEG_INFINITY;
    for m in &coups {
        let fille = pos.clone().play(m).expect("coup légal");
        let v = -negamax_materiel(&fille, depth - 1, ply + 1, -beta, -alpha);
        if v > best {
            best = v;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Évalue chaque coup racine en fenêtre pleine (pour conserver tous les ex æquo)
/// et tire au sort parmi les meilleurs.
fn choix_racine<F>(pos: &Chess, rng: &mut StdRng, mut eval_fille: F) -> Option<Move>
where
    F: FnMut(&Chess) -> f32,
{
    let coups = pos.legal_moves();
    if coups.is_empty() {
        return None;
    }
    let mut best = f32::NEG_INFINITY;
    let mut meilleurs: Vec<Move> = Vec::new();
    for m in &coups {
        let fille = pos.clone().play(m).expect("coup légal");
        let v = -eval_fille(&fille);
        if v > best + EPS_EGALITE {
            best = v;
            meilleurs.clear();
            meilleurs.push(m.clone());
        } else if (v - best).abs() <= EPS_EGALITE {
            meilleurs.push(m.clone());
        }
    }
    meilleurs.choose(rng).cloned()
}

/// Négamax alpha-bêta sur `material_eval`, profondeur `depth` (2 par défaut),
/// départage aléatoire des coups à égalité.
pub struct MaterialBot {
    pub rng: StdRng,
    pub depth: u32,
}

impl MaterialBot {
    pub fn new(seed: u64, depth: u32) -> Self {
        MaterialBot { rng: StdRng::seed_from_u64(seed), depth }
    }
}

impl Bot for MaterialBot {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        let d = self.depth.max(1);
        choix_racine(pos, &mut self.rng, |fille| {
            negamax_materiel(fille, d - 1, 1, f32::NEG_INFINITY, f32::INFINITY)
        })
    }
}

/// Négamax avec le réseau aux feuilles (perspective du trait, [-1,1]).
/// Mat/pat exacts (mêmes scores ±(1000-ply) qui dominent les valeurs réseau),
/// prises triées d'abord pour l'élagage.
fn negamax_reseau(
    net: &Mlp,
    pos: &Chess,
    depth: u32,
    ply: i32,
    mut alpha: f32,
    beta: f32,
    buf: &mut [f32],
) -> f32 {
    // Même ordre que negamax_materiel : mat/pat avant la règle des 50 coups.
    let mut coups = pos.legal_moves();
    if coups.is_empty() {
        return if pos.is_check() { -(1000.0 - ply as f32) } else { 0.0 };
    }
    if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
        return 0.0;
    }
    if depth == 0 {
        encode(pos, buf);
        return net.forward_one(buf);
    }
    coups.sort_unstable_by_key(|m| !m.is_capture());
    let mut best = f32::NEG_INFINITY;
    for m in &coups {
        let fille = pos.clone().play(m).expect("coup légal");
        let v = -negamax_reseau(net, &fille, depth - 1, ply + 1, -beta, -alpha, buf);
        if v > best {
            best = v;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break;
        }
    }
    best
}

/// Bot piloté par le réseau de valeur.
/// - `temperature > 0` (entraînement) : 1 pli — on encode chaque position fille,
///   valeur = -V(fille) (perspective adverse), échantillonnage softmax(valeurs/T).
/// - `temperature == 0` (jeu sérieux) : négamax profondeur `depth` avec le réseau
///   aux feuilles ; mat/pat détectés exactement ; les feuilles bruyantes (prise
///   possible) peuvent être stabilisées par `material_eval`.
pub struct NetBot<'a> {
    pub net: &'a Mlp,
    pub rng: StdRng,
    pub temperature: f32,
    pub depth: u32,
}

impl<'a> NetBot<'a> {
    pub fn new(net: &'a Mlp, seed: u64, temperature: f32, depth: u32) -> Self {
        NetBot { net, rng: StdRng::seed_from_u64(seed), temperature, depth }
    }
}

impl<'a> Bot for NetBot<'a> {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        let coups = pos.legal_moves();
        if coups.is_empty() {
            return None;
        }
        let mut buf = vec![0.0f32; N_FEATURES];
        if self.temperature > 0.0 {
            // 1 pli : valeur de chaque fille vue de NOTRE camp = -V(fille),
            // car la fille est évaluée du point de vue du camp adverse.
            let mut vals = Vec::with_capacity(coups.len());
            for m in &coups {
                let fille = pos.clone().play(m).expect("coup légal");
                let v = if fille.is_checkmate() {
                    1.0 // on vient de mater : gain certain
                } else if fille.is_stalemate()
                    || fille.is_insufficient_material()
                    || fille.halfmoves() >= 100
                {
                    0.0 // nulle certaine
                } else {
                    encode(&fille, &mut buf);
                    -self.net.forward_one(&buf)
                };
                vals.push(v);
            }
            // Échantillonnage softmax(valeurs / T), stabilisé par le max.
            let t = self.temperature.max(1e-6);
            let vmax = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let poids: Vec<f32> = vals.iter().map(|v| ((v - vmax) / t).exp()).collect();
            let somme: f32 = poids.iter().sum();
            let mut tirage = self.rng.gen::<f32>() * somme;
            for (i, w) in poids.iter().enumerate() {
                tirage -= w;
                if tirage <= 0.0 {
                    return Some(coups[i].clone());
                }
            }
            // Filet de sécurité numérique : dernier coup.
            coups.last().cloned()
        } else {
            // Jeu sérieux : négamax profondeur depth, réseau aux feuilles.
            let d = self.depth.max(1);
            let net = self.net;
            choix_racine(pos, &mut self.rng, |fille| {
                negamax_reseau(net, fille, d - 1, 1, f32::NEG_INFINITY, f32::INFINITY, &mut buf)
            })
        }
    }
}
