//! Adversaires : bot aléatoire, bot matériel (alpha-bêta), bot réseau,
//! bot recherche (négamax complet du module `search`).
//! Tous renvoient None uniquement s'il n'existe aucun coup légal.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use shakmaty::{Board, Chess, Color, Move, Position};

use crate::nn::{evalue_position, Mlp};
use crate::search;
use crate::syzygy;

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
/// prises triées d'abord pour l'élagage. Les feuilles passent par
/// `nn::evalue_position` : le réseau peut être de N'IMPORTE quel schéma
/// (dense 773 historique ou creux roi-zones), `buf` n'est utilisé que par le
/// chemin dense (redimensionné au besoin).
fn negamax_reseau(
    net: &Mlp,
    pos: &Chess,
    depth: u32,
    ply: i32,
    mut alpha: f32,
    beta: f32,
    buf: &mut Vec<f32>,
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
        return evalue_position(net, pos, buf);
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

/// Bot piloté par le réseau de valeur (schéma dense 773 OU creux roi-zones,
/// routé par `nn::evalue_position`).
/// - `temperature > 0` (entraînement) : 1 pli — on évalue chaque position fille,
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
        // Tampon d'encodage du chemin dense (dimensionné par evalue_position ;
        // inutilisé par le chemin creux roi-zones).
        let mut buf: Vec<f32> = Vec::new();
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
                    -evalue_position(self.net, &fille, &mut buf)
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

/// Échantillonne un coup parmi les scores racine par softmax(score / T).
/// Les scores sont clampés à [-2, 2] AVANT le softmax : un score de mat
/// (±SCORE_MAT) ne doit pas écraser la distribution — un mat « vaut » 2.
/// Softmax stabilisé par soustraction du max. None si `scores` est vide.
/// (`pub` : réutilisé par src/bin/calibration.rs pour reproduire le tirage de
/// coups du self-play d'entraînement.)
pub fn echantillonne_scores_racine(
    scores: &[(Move, f32)],
    temperature: f32,
    rng: &mut StdRng,
) -> Option<Move> {
    if scores.is_empty() {
        return None;
    }
    let t = temperature.max(1e-6);
    let clamps: Vec<f32> = scores.iter().map(|(_, s)| s.clamp(-2.0, 2.0)).collect();
    let vmax = clamps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let poids: Vec<f32> = clamps.iter().map(|v| ((v - vmax) / t).exp()).collect();
    let somme: f32 = poids.iter().sum();
    let mut tirage = rng.gen::<f32>() * somme;
    for (i, w) in poids.iter().enumerate() {
        tirage -= w;
        if tirage <= 0.0 {
            return Some(scores[i].0.clone());
        }
    }
    // Filet de sécurité numérique : dernier coup.
    scores.last().map(|(m, _)| m.clone())
}

/// Vrai si `pos` est exactement la position initiale des échecs.
fn est_position_initiale(pos: &Chess) -> bool {
    pos.fullmoves().get() == 1
        && pos.turn() == Color::White
        && pos.halfmoves() == 0
        && pos.board() == &Board::default()
}

/// Bot piloté par la recherche complète (`search::Recherche`) : alpha-bêta à
/// approfondissement itératif, TT persistante entre les coups d'une même
/// partie. `temperature == 0` → meilleur coup ; `temperature > 0` → softmax
/// sur les scores racine (clampés, voir `echantillonne_scores_racine`).
///
/// Deux armements optionnels, tous deux INACTIFS par défaut (comportement
/// historique bit à bit) et posés par les fabriques de bots : l'évaluation
/// quantizée int8 (`avec_int8`) et les tables de finales Syzygy
/// (`avec_syzygy`).
pub struct BotRecherche {
    /// UN chercheur persistant : la TT est réutilisée entre coups (gros gain),
    /// et vidée au début de chaque partie via `nouvelle_partie()`.
    recherche: search::Recherche,
    rng: StdRng,
    limites: search::Limites,
    temperature: f32,
}

impl BotRecherche {
    /// Taille de la table de transposition : 2^20 ≈ 1M d'entrées.
    const TAILLE_TT_LOG2: u32 = 20;

    pub fn new(net: Arc<Mlp>, seed: u64, limites: search::Limites, temperature: f32) -> Self {
        BotRecherche {
            recherche: search::Recherche::new(net, Self::TAILLE_TT_LOG2),
            rng: StdRng::seed_from_u64(seed),
            limites,
            temperature,
        }
    }

    /// Active (ou coupe) l'évaluation quantizée int8 de la recherche
    /// (`Recherche::utilise_int8`, défaut false — voir src/quant.rs).
    /// Consommant, pour s'enchaîner à `new` dans les fabriques de bots.
    pub fn avec_int8(mut self, actif: bool) -> Self {
        self.recherche.utilise_int8 = actif;
        self
    }

    /// Arme le bot des tables de finales Syzygy 3-4-5 (voir src/syzygy.rs) :
    /// racine ≤ 5 pièces jouée par DTZ, sondes WDL exactes dans l'arbre.
    /// `None` (le défaut de `new`) = comportement historique STRICT.
    ///
    /// L'état vit dans le champ `Recherche::syzygy` du chercheur possédé —
    /// même conception que `avec_int8`, qui écrit dans `utilise_int8` : un
    /// champ jumeau sur le bot ne ferait que dupliquer la source de vérité et
    /// pourrait diverger d'elle. Le paramètre est un `Option<Arc<_>>` (et non
    /// un `Arc`) parce que TOUS les appelants tiennent déjà l'option issue de
    /// leur drapeau `--syzygy` : `.avec_syzygy(tables.clone())` s'écrit sans
    /// branchement, et cloner l'Arc ne recharge JAMAIS les 290 fichiers.
    /// Consommant, pour s'enchaîner à `new` dans les fabriques de bots.
    pub fn avec_syzygy(mut self, tables: Option<Arc<syzygy::Tables>>) -> Self {
        self.recherche.syzygy = tables;
        self
    }
}

impl Bot for BotRecherche {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        // Détection du premier coup d'une partie : la position initiale ne
        // peut réapparaître en cours de partie (fullmoves croît), on peut donc
        // vider TT/killers/historique sans risque de faux positif.
        if est_position_initiale(pos) {
            self.recherche.nouvelle_partie();
        }
        let res = self.recherche.cherche(pos, self.limites);
        if self.temperature > 0.0 {
            if let Some(m) =
                echantillonne_scores_racine(&res.scores_racine, self.temperature, &mut self.rng)
            {
                return Some(m);
            }
        }
        res.coup
    }
}

#[cfg(test)]
mod tests_recherche {
    use super::*;

    /// Le clamp à [-2, 2] empêche un score de mat (±SCORE_MAT) d'écraser le
    /// softmax : à T = 1, les autres coups restent échantillonnés.
    #[test]
    fn echantillonnage_mat_ne_crase_pas() {
        let coups = Chess::default().legal_moves();
        assert!(coups.len() >= 2);
        // Premier coup : score de mat (+SCORE_MAT), les autres : 0.
        let scores: Vec<(Move, f32)> = coups
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), if i == 0 { crate::search::SCORE_MAT } else { 0.0 }))
            .collect();
        let mut rng = StdRng::seed_from_u64(1);
        let mut vu_mat = false;
        let mut vu_autre = false;
        for _ in 0..300 {
            let m = echantillonne_scores_racine(&scores, 1.0, &mut rng).unwrap();
            if m == scores[0].0 {
                vu_mat = true;
            } else {
                vu_autre = true;
            }
        }
        assert!(vu_mat, "le coup de mat doit rester le plus probable");
        assert!(vu_autre, "les autres coups ne doivent pas être écrasés (clamp à 2)");
    }

    /// À température quasi nulle, le meilleur coup est (quasi) toujours choisi.
    #[test]
    fn echantillonnage_froid_choisit_le_meilleur() {
        let coups = Chess::default().legal_moves();
        let scores: Vec<(Move, f32)> = coups
            .iter()
            .enumerate()
            .map(|(i, m)| (m.clone(), if i == 3 { 1.5 } else { -0.5 }))
            .collect();
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..100 {
            let m = echantillonne_scores_racine(&scores, 0.01, &mut rng).unwrap();
            assert_eq!(m, scores[3].0);
        }
    }

    /// Liste vide → None (aucun coup légal).
    #[test]
    fn echantillonnage_vide() {
        let mut rng = StdRng::seed_from_u64(3);
        assert!(echantillonne_scores_racine(&[], 0.5, &mut rng).is_none());
    }
}

/// Tables Syzygy portées par le bot (R4) : propagation du champ jusqu'à la
/// recherche, et conversion effective d'une finale gagnée.
#[cfg(test)]
mod tests_syzygy {
    use super::*;
    use crate::features::N_FEATURES;
    use crate::search::Limites;
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;

    /// Dossier des tables (relatif à la racine du crate = cwd de cargo test).
    const DOSSIER: &str = "engines/syzygy";

    /// Réseau linéaire nul (éval constante 0.0) : le test mesure les tables et
    /// la recherche, pas la qualité d'un réseau — et reste rapide en profil dev.
    fn reseau_nul() -> Arc<Mlp> {
        Arc::new(Mlp {
            sizes: vec![N_FEATURES, 1],
            weights: vec![vec![0.0; N_FEATURES]],
            biases: vec![vec![0.0]],
            adam_mw: vec![vec![0.0; N_FEATURES]],
            adam_vw: vec![vec![0.0; N_FEATURES]],
            adam_mb: vec![vec![0.0]],
            adam_vb: vec![vec![0.0]],
            steps: 0,
            pas_colonnes: vec![0u64; N_FEATURES],
        })
    }

    fn limites_noeuds(n: u64) -> Limites {
        Limites { max_noeuds: n, max_profondeur: 0, movetime_ms: 0 }
    }

    /// Joue les DEUX camps avec le même bot depuis `fen` et rend le nombre de
    /// plis jusqu'au mat, ou None si aucun mat n'est délivré en `plis_max`.
    fn plis_jusqu_au_mat(bot: &mut BotRecherche, fen: &str, plis_max: u32) -> Option<u32> {
        let mut pos: Chess = fen
            .parse::<Fen>()
            .expect("FEN valide")
            .into_position(CastlingMode::Standard)
            .expect("position légale");
        let mut plis = 0u32;
        while plis < plis_max {
            if pos.legal_moves().is_empty() {
                return pos.is_check().then_some(plis);
            }
            let coup = bot.choose(&pos)?;
            pos = pos.play(&coup).expect("coup légal");
            plis += 1;
        }
        None
    }

    /// Propagation du champ jusqu'à la recherche — sans aucun fichier de
    /// table (Tablebase vide) : ce test tourne PARTOUT, y compris sur une
    /// machine sans tables installées.
    #[test]
    fn avec_syzygy_propage_jusqu_a_la_recherche() {
        let bot = BotRecherche::new(reseau_nul(), 1, limites_noeuds(64), 0.0);
        assert!(bot.recherche.syzygy.is_none(), "défaut : aucune table (historique)");
        let bot = bot.avec_syzygy(Some(Arc::new(syzygy::Tables::new())));
        assert!(bot.recherche.syzygy.is_some(), "les tables doivent atteindre la recherche");
        // None remet le bot dans son état historique (pas d'effet cliquet).
        let bot = bot.avec_syzygy(None);
        assert!(bot.recherche.syzygy.is_none());
    }

    /// KQvK, budget de recherche volontairement famélique (400 nœuds) : ARMÉ
    /// des tables, le bot mate par DTZ ; SANS elles, le même bot (même réseau
    /// nul, mêmes limites) ne convertit pas — ou pas aussi vite. C'est
    /// exactement l'écart que R4 supprime au gating, aux ancres et au plateau.
    /// Sauté proprement si les tables ne sont pas installées.
    #[test]
    fn kqvk_les_tables_convertissent_ce_que_le_bot_seul_rate() {
        if !std::path::Path::new(DOSSIER).is_dir() {
            println!("tables absentes ({DOSSIER}) : test sauté");
            return;
        }
        let (tables, _) = syzygy::charge(DOSSIER).expect("chargement des tables");
        let tables = Arc::new(tables);
        const KQVK: &str = "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1";
        const PLIS_MAX: u32 = 60;

        let mut arme = BotRecherche::new(reseau_nul(), 7, limites_noeuds(400), 0.0)
            .avec_syzygy(Some(tables));
        let avec = plis_jusqu_au_mat(&mut arme, KQVK, PLIS_MAX)
            .expect("armé des tables, le mat KQvK doit tomber en moins de 60 plis");

        let mut nu = BotRecherche::new(reseau_nul(), 7, limites_noeuds(400), 0.0);
        let sans = plis_jusqu_au_mat(&mut nu, KQVK, PLIS_MAX);

        println!(
            "KQvK a 400 noeuds : {avec} plis AVEC les tables, {} SANS",
            match sans {
                Some(p) => format!("{p} plis"),
                None => format!("pas de mat en {PLIS_MAX}"),
            }
        );
        match sans {
            None => {} // le bot nu ne convertit pas du tout : le cas nominal.
            Some(p) => assert!(
                avec < p,
                "les tables doivent convertir plus vite : {avec} plis avec, {p} sans"
            ),
        }
    }
}
