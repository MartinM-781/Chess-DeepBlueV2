//! Évaluation incrémentale du réseau de valeur (schéma NNUE).
//!
//! Le forward complet coûte ~0,3 ms, dont ~99 % dans la couche 773→512. Or un
//! coup ne change que 2 à 4 features de pièces : on maintient donc les
//! PRÉ-ACTIVATIONS de la couche 1 (un accumulateur de 512 f32) par deltas de
//! colonnes, et seules les couches supérieures (512→64→1, ~35k multiplications)
//! sont recalculées à chaque évaluation.
//!
//! PIÈGE CENTRAL : nos features sont en perspective du TRAIT, qui alterne à
//! chaque coup — un accumulateur unique serait invalidé par tout coup joué.
//! Solution NNUE standard : DEUX accumulateurs par étage de pile, l'un en
//! perspective blanche (« comme si les blancs étaient au trait »), l'autre en
//! perspective noire. Chacun est mis à jour incrémentalement dans SA
//! perspective, et `evalue` lit celui du camp au trait de la position évaluée.
//!
//! Les 5 features de drapeaux (4 droits de roque + en passant) ne sont PAS
//! incrémentales : elles dépendent du trait et changent de sens à chaque coup.
//! Elles sont ajoutées au moment de l'évaluation (≤ 5 colonnes de 512,
//! négligeable), lues directement de la position évaluée.

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role, Square};

use crate::features::N_FEATURES;
use crate::nn::Mlp;

/// Début des 5 features de drapeaux (après les 12 plans pièce×case).
const BASE_DRAPEAUX: usize = 12 * 64;

/// Une couche dense au-dessus de l'accumulateur (poids row-major sortie×entrée,
/// même convention que `Mlp` pour reproduire exactement ses boucles).
struct CoucheSup {
    n_in: usize,
    n_out: usize,
    poids: Vec<f32>,
    biais: Vec<f32>,
}

/// Poids du réseau réorganisés pour l'évaluation incrémentale.
/// Construit UNE FOIS depuis un `Mlp` (qui reste la source de vérité) :
/// la couche 1 est stockée TRANSPOSÉE (une colonne de `h1` f32 contiguë par
/// feature) pour que l'ajout/retrait d'une feature soit un parcours linéaire.
pub struct EvalIncrementale {
    /// Largeur de la couche 1 (512 pour le réseau réel).
    h1: usize,
    /// Colonnes de la couche 1, à plat : colonne de la feature f =
    /// `cols[f*h1 .. (f+1)*h1]`.
    cols: Vec<f32>,
    /// Biais de la couche 1 (placés dans l'accumulateur initial).
    biais1: Vec<f32>,
    /// Couches au-dessus de l'accumulateur (512→64→1 pour le réseau réel).
    sup: Vec<CoucheSup>,
}

impl EvalIncrementale {
    /// Copie les poids de `net` dans la disposition incrémentale.
    /// Le `Mlp` n'est pas modifié et peut continuer à être entraîné ; il faut
    /// alors reconstruire un `EvalIncrementale` pour voir les nouveaux poids.
    pub fn new(net: &Mlp) -> Self {
        assert!(
            net.sizes.len() >= 3,
            "EvalIncrementale: il faut au moins entrée → cachée → sortie"
        );
        assert_eq!(
            net.sizes[0], N_FEATURES,
            "EvalIncrementale: la couche d'entrée doit faire N_FEATURES"
        );
        let h1 = net.sizes[1];
        assert_eq!(net.weights[0].len(), h1 * N_FEATURES);

        // Transposition de la couche 1 : Mlp range w1[j*773 + f] (ligne par
        // neurone j), on veut cols[f*h1 + j] (colonne par feature f).
        let w1 = &net.weights[0];
        let mut cols = vec![0.0f32; N_FEATURES * h1];
        for j in 0..h1 {
            let ligne = &w1[j * N_FEATURES..(j + 1) * N_FEATURES];
            for f in 0..N_FEATURES {
                cols[f * h1 + j] = ligne[f];
            }
        }

        // Couches supérieures copiées telles quelles (row-major, comme Mlp).
        let sup = (1..net.sizes.len() - 1)
            .map(|l| CoucheSup {
                n_in: net.sizes[l],
                n_out: net.sizes[l + 1],
                poids: net.weights[l].clone(),
                biais: net.biases[l].clone(),
            })
            .collect();

        EvalIncrementale { h1, cols, biais1: net.biases[0].clone(), sup }
    }

    /// Colonne de la couche 1 associée à la feature `f`.
    #[inline]
    fn colonne(&self, f: usize) -> &[f32] {
        &self.cols[f * self.h1..(f + 1) * self.h1]
    }

    /// Encode complètement `pos` dans les DEUX perspectives : c'est la racine
    /// de la pile d'accumulateurs (biais de couche 1 + colonnes des pièces).
    pub fn racine(&self, pos: &Chess) -> PileAccus {
        let h1 = self.h1;
        // Réserve pour ~128 plis de recherche sans réallocation.
        let mut donnees = Vec::with_capacity(2 * h1 * 128);
        donnees.extend_from_slice(&self.biais1);
        donnees.extend_from_slice(&self.biais1);
        {
            let (blanc, noir) = donnees.split_at_mut(h1);
            for (case, piece) in pos.board().iter() {
                let (ib, inoir) = indices_piece(piece.color, piece.role, case);
                accumule(blanc, self.colonne(ib), 1.0);
                accumule(noir, self.colonne(inoir), 1.0);
            }
        }
        PileAccus { donnees, h1 }
    }
}

/// Indices de la feature d'une pièce (couleur, rôle, case) dans CHAQUE
/// perspective : (perspective blanche, perspective noire).
/// Convention EXACTE de `features::encode` : pour la perspective P,
/// plan = role-1 si la pièce est du camp P, sinon 6 + role-1, et la case est
/// vue par P (miroir `case ^ 56` pour la perspective noire).
#[inline]
fn indices_piece(couleur: Color, role: Role, case: Square) -> (usize, usize) {
    let r = usize::from(role) - 1;
    let c = usize::from(case);
    let plan_blanc = if couleur == Color::White { r } else { 6 + r };
    let plan_noir = if couleur == Color::Black { r } else { 6 + r };
    (plan_blanc * 64 + c, plan_noir * 64 + (c ^ 56))
}

/// `dst += signe * col`, élément par élément (signe ∈ {+1, -1}, exact en f32).
#[inline]
fn accumule(dst: &mut [f32], col: &[f32], signe: f32) {
    debug_assert_eq!(dst.len(), col.len());
    for (d, c) in dst.iter_mut().zip(col) {
        *d += signe * *c;
    }
}

/// Pile d'accumulateurs de la couche 1, un étage par pli de recherche.
/// Chaque étage contient 2×h1 f32 : [perspective blanche | perspective noire].
/// `pousse` duplique le sommet puis applique les deltas du coup ; `depousse`
/// revient à l'étage précédent sans aucun recalcul.
pub struct PileAccus {
    /// Étages concaténés : l'étage k occupe `donnees[k*2*h1 .. (k+1)*2*h1]`.
    donnees: Vec<f32>,
    h1: usize,
}

impl PileAccus {
    /// Tranche du sommet de pile (2×h1 valeurs).
    #[inline]
    fn base_sommet(&self) -> usize {
        self.donnees.len() - 2 * self.h1
    }

    /// Empile la position atteinte en jouant `m` depuis `pos_avant` (position
    /// AVANT le coup, dont le trait est le camp qui joue). Seules les colonnes
    /// des 2 à 4 features modifiées sont touchées, dans les DEUX perspectives.
    pub fn pousse(&mut self, eval: &EvalIncrementale, pos_avant: &Chess, m: &Move) {
        debug_assert_eq!(self.h1, eval.h1, "pousse: EvalIncrementale d'une autre taille");
        let h1 = self.h1;
        let base = self.base_sommet();
        // Duplique le sommet : le nouvel étage part de la position courante.
        self.donnees.extend_from_within(base..);
        let sommet = self.donnees.len() - 2 * h1;
        let (blanc, noir) = self.donnees[sommet..].split_at_mut(h1);

        let nous = pos_avant.turn();
        // ±colonne d'une pièce, appliqué aux deux perspectives d'un coup.
        let mut delta = |couleur: Color, role: Role, case: Square, signe: f32| {
            let (ib, inoir) = indices_piece(couleur, role, case);
            accumule(blanc, eval.colonne(ib), signe);
            accumule(noir, eval.colonne(inoir), signe);
        };

        match m {
            Move::Normal { role, from, capture, to, promotion } => {
                delta(nous, *role, *from, -1.0);
                if let Some(prise) = capture {
                    // Capture normale : la victime est sur la case d'arrivée.
                    delta(nous.other(), *prise, *to, -1.0);
                }
                // Promotion : le pion disparaît de `from`, la pièce promue
                // apparaît sur `to`.
                delta(nous, promotion.unwrap_or(*role), *to, 1.0);
            }
            Move::EnPassant { from, to } => {
                // ATTENTION : le pion pris n'est PAS sur la case d'arrivée,
                // mais sur (colonne de `to`, rangée de `from`).
                delta(nous, Role::Pawn, *from, -1.0);
                delta(
                    nous.other(),
                    Role::Pawn,
                    Square::from_coords(to.file(), from.rank()),
                    -1.0,
                );
                delta(nous, Role::Pawn, *to, 1.0);
            }
            Move::Castle { king, rook } => {
                // Convention shakmaty : `Move::to()` est la case de la TOUR ;
                // ici on destructure directement les cases d'origine du roi et
                // de la tour, et les cases d'arrivée réelles viennent du côté
                // de roque (g1/f1, c1/d1, etc. selon la couleur).
                let cote = m.castling_side().expect("Move::Castle a toujours un côté");
                delta(nous, Role::King, *king, -1.0);
                delta(nous, Role::Rook, *rook, -1.0);
                delta(nous, Role::King, cote.king_to(nous), 1.0);
                delta(nous, Role::Rook, cote.rook_to(nous), 1.0);
            }
            Move::Put { .. } => unreachable!("Move::Put n'existe qu'en Crazyhouse"),
        }
    }

    /// Null-move : la position est inchangée, seul le trait s'inverse — les
    /// accumulateurs sont donc identiques (l'évaluation lira simplement
    /// l'autre perspective). On duplique le sommet pour garder la symétrie
    /// pousse/depousse de la recherche.
    pub fn pousse_null(&mut self) {
        let base = self.base_sommet();
        self.donnees.extend_from_within(base..);
    }

    /// Dépile un étage (retour à la position précédente, coût nul).
    pub fn depousse(&mut self) {
        assert!(
            self.donnees.len() >= 4 * self.h1,
            "depousse: la racine de la pile ne peut pas être dépilée"
        );
        let nouvelle_taille = self.donnees.len() - 2 * self.h1;
        self.donnees.truncate(nouvelle_taille);
    }

    /// Évalue la position du sommet de pile (`pos` DOIT être cette position) :
    /// lit l'accumulateur de la perspective du trait, ajoute les colonnes des
    /// drapeaux roques/en passant actifs, applique ReLU puis les couches
    /// supérieures et tanh. Égal à `net.forward_one(encode(pos))` à ~1e-4
    /// (seul l'ordre des sommations f32 de la couche 1 diffère).
    pub fn evalue(&self, eval: &EvalIncrementale, pos: &Chess) -> f32 {
        debug_assert_eq!(self.h1, eval.h1, "evalue: EvalIncrementale d'une autre taille");
        let h1 = self.h1;
        let base = self.base_sommet();
        let nous = pos.turn();
        let sommet = &self.donnees[base..];
        let accu = if nous == Color::White { &sommet[..h1] } else { &sommet[h1..] };

        // Copie de travail : le sommet de pile ne doit pas être modifié.
        let mut courant: Vec<f32> = accu.to_vec();

        // Drapeaux non incrémentaux, mêmes conditions que `features::encode` :
        // notre O-O, notre O-O-O, leur O-O, leur O-O-O, en passant légal.
        let eux = nous.other();
        let roques = pos.castles();
        if roques.has(nous, CastlingSide::KingSide) {
            accumule(&mut courant, eval.colonne(BASE_DRAPEAUX), 1.0);
        }
        if roques.has(nous, CastlingSide::QueenSide) {
            accumule(&mut courant, eval.colonne(BASE_DRAPEAUX + 1), 1.0);
        }
        if roques.has(eux, CastlingSide::KingSide) {
            accumule(&mut courant, eval.colonne(BASE_DRAPEAUX + 2), 1.0);
        }
        if roques.has(eux, CastlingSide::QueenSide) {
            accumule(&mut courant, eval.colonne(BASE_DRAPEAUX + 3), 1.0);
        }
        if pos.ep_square(EnPassantMode::Legal).is_some() {
            accumule(&mut courant, eval.colonne(BASE_DRAPEAUX + 4), 1.0);
        }

        // ReLU de la couche 1 (les pré-activations deviennent des activations).
        for v in courant.iter_mut() {
            *v = v.max(0.0);
        }

        // Couches supérieures : mêmes boucles que `Mlp::avancer` (row-major,
        // même ordre de sommation → mêmes arrondis sur cette partie).
        let n_sup = eval.sup.len();
        let mut suivant: Vec<f32> = Vec::new();
        for (l, couche) in eval.sup.iter().enumerate() {
            suivant.clear();
            suivant.resize(couche.n_out, 0.0);
            for j in 0..couche.n_out {
                let ligne = &couche.poids[j * couche.n_in..(j + 1) * couche.n_in];
                let mut s = couche.biais[j];
                for k in 0..couche.n_in {
                    s += ligne[k] * courant[k];
                }
                // ReLU sur les couches cachées, tanh en sortie.
                suivant[j] = if l + 1 == n_sup { s.tanh() } else { s.max(0.0) };
            }
            std::mem::swap(&mut courant, &mut suivant);
        }
        courant[0]
    }
}

// ---------------------------------------------------------------------------
// Batterie de PARITÉ : l'évaluation incrémentale doit être indiscernable du
// forward complet (à ~1e-4) sur positions statiques, parties entières (roques,
// promotions, en passant), null-move et séquences pousse/depousse.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{Bot, RandomBot};
    use crate::features::encode;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::{CastlingMode, FromSetup};

    /// Tolérance de parité : seul l'ordre des sommations f32 diffère.
    const TOL: f32 = 1e-4;

    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    /// Référence : encodage complet + forward complet du Mlp.
    fn reference(net: &Mlp, pos: &Chess) -> f32 {
        let mut buf = vec![0.0f32; N_FEATURES];
        encode(pos, &mut buf);
        net.forward_one(&buf)
    }

    /// Petit Mlp aléatoire [773, h1, h2, 1] construit par les champs publics :
    /// mêmes conventions que `Mlp::new` mais avec des BIAIS NON NULS (un bug
    /// qui perdrait les biais passerait inaperçu avec les biais à zéro de
    /// `Mlp::new`) et des couches étroites pour des tests rapides en debug.
    fn petit_reseau(seed: u64, h1: usize, h2: usize) -> Mlp {
        let sizes = vec![N_FEATURES, h1, h2, 1];
        let mut rng = StdRng::seed_from_u64(seed);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for l in 0..sizes.len() - 1 {
            let (n_in, n_out) = (sizes[l], sizes[l + 1]);
            let ecart = (2.0 / n_in as f32).sqrt();
            weights.push(
                (0..n_in * n_out)
                    .map(|_| (rng.gen::<f32>() * 2.0 - 1.0) * ecart)
                    .collect::<Vec<f32>>(),
            );
            biases.push((0..n_out).map(|_| rng.gen::<f32>() * 0.2 - 0.1).collect::<Vec<f32>>());
        }
        let zw: Vec<Vec<f32>> = weights.iter().map(|w| vec![0.0; w.len()]).collect();
        let zb: Vec<Vec<f32>> = biases.iter().map(|b| vec![0.0; b.len()]).collect();
        Mlp {
            sizes,
            weights,
            biases,
            adam_mw: zw.clone(),
            adam_vw: zw,
            adam_mb: zb.clone(),
            adam_vb: zb,
            steps: 0,
        }
    }

    /// Partie RandomBot depuis `depart` : liste (position AVANT coup, coup),
    /// arrêtée à la fin de partie (mêmes conditions que selfplay/arena) ou à
    /// `max_plies`.
    fn partie_aleatoire(depart: &Chess, seed: u64, max_plies: usize) -> Vec<(Chess, Move)> {
        let mut bot = RandomBot::new(seed);
        let mut pos = depart.clone();
        let mut partie = Vec::new();
        for _ in 0..max_plies {
            if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
                break;
            }
            let m = match bot.choose(&pos) {
                Some(m) => m,
                None => break, // mat ou pat
            };
            let suivante = pos.clone().play(&m).expect("coup légal");
            partie.push((pos, m));
            pos = suivante;
        }
        partie
    }

    /// Partie scriptée en notation UCI depuis `depart`.
    fn construit_partie(depart: &Chess, ucis: &[&str]) -> Vec<(Chess, Move)> {
        let mut pos = depart.clone();
        let mut partie = Vec::new();
        for u in ucis {
            let m = UciMove::from_ascii(u.as_bytes())
                .expect("UCI invalide")
                .to_move(&pos)
                .expect("coup illégal dans la partie construite");
            let suivante = pos.clone().play(&m).expect("coup légal");
            partie.push((pos, m));
            pos = suivante;
        }
        partie
    }

    /// Rejoue `partie` en maintenant la pile : parité exigée APRÈS CHAQUE coup.
    fn verifie_partie(
        net: &Mlp,
        eval: &EvalIncrementale,
        partie: &[(Chess, Move)],
        contexte: &str,
    ) {
        if partie.is_empty() {
            return;
        }
        let mut pile = eval.racine(&partie[0].0);
        for (i, (avant, m)) in partie.iter().enumerate() {
            pile.pousse(eval, avant, m);
            let apres = avant.clone().play(m).expect("coup légal");
            let attendu = reference(net, &apres);
            let obtenu = pile.evalue(eval, &apres);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "{contexte}, coup {i} ({m:?}) : incrémental {obtenu} vs référence {attendu}"
            );
        }
    }

    /// Même plateau, trait inversé (pour tester le null-move) ; None si la
    /// position obtenue est illégale (roi de l'ancien trait en prise).
    fn inverse_trait(pos: &Chess) -> Option<Chess> {
        let mut setup = pos.clone().into_setup(EnPassantMode::Legal);
        setup.turn = !setup.turn;
        setup.ep_square = None; // un null-move annule toute prise en passant
        Chess::from_setup(setup, CastlingMode::Standard).ok()
    }

    /// 1. Parité statique : 500 positions de parties RandomBot, réseau aux
    /// tailles RÉELLES [773,512,64,1] (biais rendus non nuls), racine().evalue()
    /// doit égaler forward_one(encode()) à 1e-4.
    #[test]
    fn parite_statique_500_positions() {
        let mut net = Mlp::new(42);
        let mut rng = StdRng::seed_from_u64(9);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        let eval = EvalIncrementale::new(&net);

        let mut positions = vec![Chess::default()];
        let mut graine = 0u64;
        while positions.len() < 500 {
            for (pos, _) in partie_aleatoire(&Chess::default(), 1000 + graine, 90) {
                positions.push(pos);
                if positions.len() >= 500 {
                    break;
                }
            }
            graine += 1;
        }

        for (i, pos) in positions.iter().enumerate() {
            let attendu = reference(&net, pos);
            let obtenu = eval.racine(pos).evalue(&eval, pos);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "position {i} : incrémental {obtenu} vs référence {attendu}"
            );
        }
    }

    /// 2a. Parité incrémentale : 200 parties aléatoires, pousse() + evalue()
    /// comparé au forward complet à CHAQUE coup. Réseau étroit (biais non nuls)
    /// pour rester rapide en debug — la logique testée est identique.
    #[test]
    fn parite_incrementale_200_parties() {
        let net = petit_reseau(3, 32, 12);
        let eval = EvalIncrementale::new(&net);
        let (mut roques, mut promotions, mut en_passants) = (0usize, 0usize, 0usize);
        for g in 0..200u64 {
            let partie = partie_aleatoire(&Chess::default(), 5000 + g, 140);
            for (_, m) in &partie {
                if m.is_castle() {
                    roques += 1;
                }
                if m.promotion().is_some() {
                    promotions += 1;
                }
                if m.is_en_passant() {
                    en_passants += 1;
                }
            }
            verifie_partie(&net, &eval, &partie, &format!("partie aléatoire {g}"));
        }
        // Couverture naturelle attendue sur 200 parties (graines fixes).
        println!("couverture : {roques} roques, {promotions} promotions, {en_passants} e.p.");
        assert!(roques > 0, "aucun roque rencontré dans les parties aléatoires");
        assert!(promotions > 0, "aucune promotion rencontrée dans les parties aléatoires");
        assert!(en_passants > 0, "aucune prise en passant rencontrée");
    }

    /// 2b. Parties construites : les QUATRE combinaisons couleur × côté de
    /// roque, chacune forcée par un coup scripté (petit blanc + grand noir,
    /// puis grand blanc + petit noir) — la couverture ne dépend d'aucun aléa.
    #[test]
    fn parite_roques_blanc_et_noir() {
        let net = petit_reseau(21, 24, 8);
        let eval = EvalIncrementale::new(&net);
        // 1.e4 d5 2.Nf3 Nc6 3.Bc4 Bf5 4.O-O Qd6 5.d3 O-O-O
        let petit_blanc_grand_noir: &[&str] = &[
            "e2e4", "d7d5", "g1f3", "b8c6", "f1c4", "c8f5", "e1g1", "d8d6", "d2d3", "e8c8",
        ];
        // 1.d4 e5 2.Be3 Be7 3.Nc3 Nf6 4.Qd2 O-O 5.O-O-O
        let grand_blanc_petit_noir: &[&str] = &[
            "d2d4", "e7e5", "c1e3", "f8e7", "b1c3", "g8f6", "d1d2", "e8g8", "e1c1",
        ];
        // vues[couleur][côté] : (blanc, noir) × (O-O, O-O-O).
        let mut vues = [[false; 2]; 2];
        for (ucis, contexte) in [
            (petit_blanc_grand_noir, "partie O-O blanc / O-O-O noir"),
            (grand_blanc_petit_noir, "partie O-O-O blanc / O-O noir"),
        ] {
            let partie = construit_partie(&Chess::default(), ucis);
            for (avant, m) in &partie {
                if let Some(cote) = m.castling_side() {
                    vues[usize::from(avant.turn() == Color::Black)]
                        [usize::from(cote == CastlingSide::QueenSide)] = true;
                }
            }
            verifie_partie(&net, &eval, &partie, contexte);
        }
        assert_eq!(
            vues,
            [[true; 2]; 2],
            "chaque combinaison couleur × côté de roque doit être exercée"
        );
    }

    /// 2c. Partie construite : promotion AVEC capture pour les deux camps
    /// (dame blanche g7xh8=Q, sous-promotion noire g2xh1=N).
    #[test]
    fn parite_promotion_avec_capture() {
        let net = petit_reseau(22, 24, 8);
        let eval = EvalIncrementale::new(&net);
        let depart = pos_de_fen("rnbqkb1r/ppppppPp/8/8/8/8/PPPPPPpP/RNBQKB1R w KQkq - 0 1");
        let partie = construit_partie(&depart, &["g7h8q", "g2h1n"]);
        assert!(
            partie
                .iter()
                .all(|(_, m)| m.promotion().is_some() && m.capture().is_some()),
            "les deux coups doivent être des promotions avec capture"
        );
        verifie_partie(&net, &eval, &partie, "partie promotions");
    }

    /// 2c bis. Partie construite : promotion CALME (sans capture) pour les
    /// deux camps (dame blanche g8=Q, sous-promotion noire g1=N) — le delta
    /// est alors -pion@from / +pièce@to, sans retrait de victime.
    #[test]
    fn parite_promotion_sans_capture() {
        let net = petit_reseau(25, 24, 8);
        let eval = EvalIncrementale::new(&net);
        // Roi noir en e7 (pas e8) : la dame promue en g8 ne donne pas échec
        // par la 8e rangée, le coup noir g2g1n reste donc jouable.
        let depart = pos_de_fen("8/4k1P1/8/8/8/8/6p1/4K3 w - - 0 1");
        let partie = construit_partie(&depart, &["g7g8q", "g2g1n"]);
        assert!(
            partie
                .iter()
                .all(|(_, m)| m.promotion().is_some() && m.capture().is_none()),
            "les deux coups doivent être des promotions sans capture"
        );
        verifie_partie(&net, &eval, &partie, "partie promotions calmes");
    }

    /// 2d. Partie construite : prise en passant blanche (e5xd6) PUIS noire
    /// (b4xc3), avec drapeau e.p. actif sur les positions intermédiaires.
    #[test]
    fn parite_en_passant_deux_camps() {
        let net = petit_reseau(23, 24, 8);
        let eval = EvalIncrementale::new(&net);
        let ucis = [
            "e2e4", "g8f6", "e4e5", "d7d5", "e5d6", "c7d6", "g1f3", "b7b5", "f3g1", "b5b4",
            "c2c4", "b4c3",
        ];
        let partie = construit_partie(&Chess::default(), &ucis);
        assert_eq!(
            partie.iter().filter(|(_, m)| m.is_en_passant()).count(),
            2,
            "la partie doit contenir une prise en passant par camp"
        );
        verifie_partie(&net, &eval, &partie, "partie en passant");
    }

    /// 3. Null-move : après pousse_null, evalue() sur la position au trait
    /// inversé doit égaler le forward complet de cette position — d'abord sur
    /// des paires de FEN fixes, puis en cours de parties aléatoires.
    #[test]
    fn pousse_null_echange_les_perspectives() {
        let net = petit_reseau(11, 24, 8);
        let eval = EvalIncrementale::new(&net);

        // Paires (trait blanc, trait noir) sur le même plateau, sans e.p.
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R",
            "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R",
        ];
        for plateau in fens {
            let pos = pos_de_fen(&format!("{plateau} w KQkq - 0 1"));
            let pos_inverse = pos_de_fen(&format!("{plateau} b KQkq - 0 1"));

            let mut pile = eval.racine(&pos);
            let origine = pile.evalue(&eval, &pos);
            pile.pousse_null();
            let obtenu = pile.evalue(&eval, &pos_inverse);
            let attendu = reference(&net, &pos_inverse);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "null-move sur {plateau} : {obtenu} vs {attendu}"
            );
            // Dépiler le null-move restitue exactement l'évaluation d'origine.
            pile.depousse();
            assert_eq!(pile.evalue(&eval, &pos), origine);
        }

        // Null-move en cours de partie : sur chaque position atteinte où
        // l'inversion du trait est légale, la parité doit tenir.
        let mut testes = 0;
        for g in 0..6u64 {
            let partie = partie_aleatoire(&Chess::default(), 300 + g, 60);
            if partie.is_empty() {
                continue;
            }
            let mut pile = eval.racine(&partie[0].0);
            for (avant, m) in &partie {
                pile.pousse(&eval, avant, m);
                let apres = avant.clone().play(m).expect("coup légal");
                if let Some(inverse) = inverse_trait(&apres) {
                    pile.pousse_null();
                    let obtenu = pile.evalue(&eval, &inverse);
                    let attendu = reference(&net, &inverse);
                    assert!(
                        (attendu - obtenu).abs() <= TOL,
                        "null-move en partie {g} : {obtenu} vs {attendu}"
                    );
                    pile.depousse();
                    testes += 1;
                }
            }
        }
        assert!(testes > 20, "trop peu de null-moves testés en partie ({testes})");
    }

    /// 4. depousse : marche aléatoire pousse/depousse (60 % / 40 %) avec pile
    /// miroir de positions ; parité avec le forward complet après CHAQUE pas.
    #[test]
    fn depousse_rejoint_la_reference() {
        let net = petit_reseau(5, 24, 8);
        let eval = EvalIncrementale::new(&net);
        let mut rng = StdRng::seed_from_u64(77);

        let mut pile = eval.racine(&Chess::default());
        let mut positions = vec![Chess::default()];
        for pas in 0..1500 {
            let sommet = positions.last().unwrap().clone();
            let coups = sommet.legal_moves();
            let pousser = !coups.is_empty() && (positions.len() == 1 || rng.gen_bool(0.6));
            if pousser {
                let m = coups[rng.gen_range(0..coups.len())].clone();
                pile.pousse(&eval, &sommet, &m);
                positions.push(sommet.play(&m).expect("coup légal"));
            } else if positions.len() > 1 {
                pile.depousse();
                positions.pop();
            } else {
                break; // partie terminée à la racine (impossible depuis l'init)
            }
            let pos = positions.last().unwrap();
            let attendu = reference(&net, pos);
            let obtenu = pile.evalue(&eval, pos);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "pas {pas} (profondeur {}) : {obtenu} vs {attendu}",
                positions.len()
            );
        }
    }

    /// Dépiler la racine est un bug de l'appelant : panique attendue.
    #[test]
    #[should_panic(expected = "depousse")]
    fn depousse_sous_la_racine_panique() {
        let net = petit_reseau(6, 8, 4);
        let eval = EvalIncrementale::new(&net);
        let mut pile = eval.racine(&Chess::default());
        pile.depousse();
    }

    /// 5. Bench (ignoré par défaut) : évals/s de evalue() contre forward_one()
    /// sur le réseau réel [773,512,64,1]. Lancer avec :
    /// `cargo test --lib nnue:: -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_evalue_contre_forward() {
        let net = Mlp::new(2);
        let eval = EvalIncrementale::new(&net);
        let pos = pos_de_fen("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 3 9");
        let coup = pos.legal_moves()[0].clone();
        let mut pile = eval.racine(&pos);
        let mut buf = vec![0.0f32; N_FEATURES];
        encode(&pos, &mut buf);
        let mut somme = 0.0f64;

        // Forward complet (sans l'encodage, mesuré à part).
        let n_fwd = 300;
        let t = std::time::Instant::now();
        for _ in 0..n_fwd {
            somme += net.forward_one(&buf) as f64;
        }
        let par_s_fwd = n_fwd as f64 / t.elapsed().as_secs_f64();

        // Encodage + forward (chemin réellement utilisé aujourd'hui).
        let t = std::time::Instant::now();
        for _ in 0..n_fwd {
            encode(&pos, &mut buf);
            somme += net.forward_one(&buf) as f64;
        }
        let par_s_enc_fwd = n_fwd as f64 / t.elapsed().as_secs_f64();

        // Évaluation incrémentale seule.
        let n_inc = 5000;
        let t = std::time::Instant::now();
        for _ in 0..n_inc {
            somme += pile.evalue(&eval, &pos) as f64;
        }
        let par_s_inc = n_inc as f64 / t.elapsed().as_secs_f64();

        // Cycle complet de recherche : pousse + evalue + depousse.
        let apres = pos.clone().play(&coup).expect("coup légal");
        let t = std::time::Instant::now();
        for _ in 0..n_inc {
            pile.pousse(&eval, &pos, &coup);
            somme += pile.evalue(&eval, &apres) as f64;
            pile.depousse();
        }
        let par_s_cycle = n_inc as f64 / t.elapsed().as_secs_f64();

        println!("forward_one seul      : {par_s_fwd:>10.0} évals/s");
        println!("encode + forward_one  : {par_s_enc_fwd:>10.0} évals/s");
        println!("evalue incrémental    : {par_s_inc:>10.0} évals/s  (×{:.1} vs forward)", par_s_inc / par_s_fwd);
        println!("pousse+evalue+depousse: {par_s_cycle:>10.0} évals/s  (×{:.1} vs encode+forward)", par_s_cycle / par_s_enc_fwd);
        assert!(somme.is_finite());
    }
}
