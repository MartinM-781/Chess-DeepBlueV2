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
//!
//! DEUX SCHÉMAS de features sont servis par la même pile (`nn::SchemaFeatures`,
//! dicté par le réseau) : le dense historique `Classique773` (ci-dessus,
//! chemin inchangé) et le creux `RoiZones8` (`features_roi`, 6149 entrées) où
//! chaque plan pièce-case est conditionné par la ZONE du roi du camp de la
//! perspective. L'accumulateur de chaque perspective est alors conditionné par
//! la zone de SON PROPRE roi : un coup ordinaire s'applique par deltas (les
//! zones ne bougent pas), mais un coup de roi qui TRAVERSE une frontière de
//! zone invalide toutes les features de pièces de SA perspective — cet
//! accumulateur est RECONSTRUIT en entier (le standard NNUE), l'autre
//! perspective gardant ses deltas. Les 5 scalaires de queue (6144..6149,
//! mêmes définitions) sont ajoutés à l'évaluation comme les drapeaux du
//! schéma classique.

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role, Square};

use crate::features::N_FEATURES;
use crate::features_roi::{zone_roi, N_FEATURES_ROI, N_ZONES_ROI};
use crate::nn::{Mlp, SchemaFeatures};

/// Début des 5 features de drapeaux (après les 12 plans pièce×case).
/// (`pub(crate)` : partagé avec le chemin quantizé de `quant.rs`.)
pub(crate) const BASE_DRAPEAUX: usize = 12 * 64;

/// Début des 5 scalaires du schéma roi-zones (après les 8 zones × 768 plans).
pub(crate) const BASE_DRAPEAUX_ROI8: usize = N_ZONES_ROI * 768;

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
    /// Schéma de features du réseau source : dicte l'indexation des colonnes
    /// (773 dense classique ou 6149 roi-zones) et le chemin de `pousse`.
    schema: SchemaFeatures,
    /// Largeur de la couche 1 (`sizes[1]` : 512 pour le réseau par défaut).
    h1: usize,
    /// Colonnes de la couche 1, à plat : colonne de la feature f =
    /// `cols[f*h1 .. (f+1)*h1]`.
    cols: Vec<f32>,
    /// Biais de la couche 1 (placés dans l'accumulateur initial).
    biais1: Vec<f32>,
    /// Couches au-dessus de l'accumulateur, en nombre ARBITRAIRE (512→64 puis
    /// 64→1 pour le réseau par défaut ; toute tête ReLU…tanh convient, ex.
    /// [1024,128,1] élargie ou [256,64,32,1] profonde).
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
        // Taille d'entrée dictée par le schéma du réseau : 773 (dense
        // classique) ou 6149 (roi-zones). La disposition transposée des
        // colonnes est identique dans les deux cas, seule leur QUANTITÉ change.
        let schema = net.schema();
        let n_in = match schema {
            SchemaFeatures::Classique773 => N_FEATURES,
            SchemaFeatures::RoiZones8 => N_FEATURES_ROI,
        };
        assert_eq!(
            net.sizes[0], n_in,
            "EvalIncrementale: la couche d'entrée doit faire N_FEATURES ({N_FEATURES}) ou N_FEATURES_ROI ({N_FEATURES_ROI})"
        );
        // Même garde que `Mlp::new_avec_tailles` : `evalue` lit `courant[0]`
        // et suppose une sortie scalaire tanh — refus immédiat et lisible.
        assert_eq!(
            *net.sizes.last().unwrap(), 1,
            "EvalIncrementale: la dernière couche doit valoir 1 (sortie scalaire tanh), reçu {:?}",
            net.sizes
        );
        let h1 = net.sizes[1];
        assert_eq!(net.weights[0].len(), h1 * n_in);

        // Transposition de la couche 1 : Mlp range w1[j*n_in + f] (ligne par
        // neurone j), on veut cols[f*h1 + j] (colonne par feature f).
        let w1 = &net.weights[0];
        let mut cols = vec![0.0f32; n_in * h1];
        for j in 0..h1 {
            let ligne = &w1[j * n_in..(j + 1) * n_in];
            for f in 0..n_in {
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

        EvalIncrementale { schema, h1, cols, biais1: net.biases[0].clone(), sup }
    }

    /// Colonne de la couche 1 associée à la feature `f`.
    #[inline]
    fn colonne(&self, f: usize) -> &[f32] {
        &self.cols[f * self.h1..(f + 1) * self.h1]
    }

    /// Encode complètement `pos` dans les DEUX perspectives : c'est la racine
    /// de la pile d'accumulateurs (biais de couche 1 + colonnes des pièces).
    /// En schéma roi-zones, chaque perspective est indexée par la zone de SON
    /// roi (conventions de `features_roi::actifs_perspective`).
    pub fn racine(&self, pos: &Chess) -> PileAccus {
        if self.schema == SchemaFeatures::RoiZones8 {
            return self.racine_roi8(pos);
        }
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

    /// `racine` du schéma roi-zones : mêmes biais + colonnes des pièces, mais
    /// chaque perspective est conditionnée par la zone de SON PROPRE roi.
    fn racine_roi8(&self, pos: &Chess) -> PileAccus {
        let h1 = self.h1;
        // Réserve pour ~128 plis de recherche sans réallocation.
        let mut donnees = Vec::with_capacity(2 * h1 * 128);
        donnees.extend_from_slice(&self.biais1);
        donnees.extend_from_slice(&self.biais1);
        {
            let (blanc, noir) = donnees.split_at_mut(h1);
            let (zone_blanche, zone_noire) = zones_rois(pos);
            for (case, piece) in pos.board().iter() {
                let ib = indice_piece_roi8(piece.color, piece.role, case, true, zone_blanche);
                let inoir = indice_piece_roi8(piece.color, piece.role, case, false, zone_noire);
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
pub(crate) fn indices_piece(couleur: Color, role: Role, case: Square) -> (usize, usize) {
    let r = usize::from(role) - 1;
    let c = usize::from(case);
    let plan_blanc = if couleur == Color::White { r } else { 6 + r };
    let plan_noir = if couleur == Color::Black { r } else { 6 + r };
    (plan_blanc * 64 + c, plan_noir * 64 + (c ^ 56))
}

/// Zones des rois pour le schéma roi-zones : (zone du roi BLANC vue de la
/// perspective blanche, zone du roi NOIR vue de la perspective noire — donc
/// après le miroir `case ^ 56`, comme dans `features_roi`).
#[inline]
pub(crate) fn zones_rois(pos: &Chess) -> (usize, usize) {
    let blanc = pos
        .board()
        .king_of(Color::White)
        .expect("position légale : roi blanc présent");
    let noir = pos
        .board()
        .king_of(Color::Black)
        .expect("position légale : roi noir présent");
    (zone_roi(usize::from(blanc)), zone_roi(usize::from(noir) ^ 56))
}

/// Indice roi-zones de la feature d'une pièce pour UNE perspective, la zone du
/// roi de CETTE perspective étant déjà connue. Convention EXACTE de
/// `features_roi::actifs_perspective` : `zone·768 + plan·64 + case_vue`, avec
/// miroir `case ^ 56` et échange des couleurs pour la perspective noire.
#[inline]
pub(crate) fn indice_piece_roi8(
    couleur: Color,
    role: Role,
    case: Square,
    perspective_blanche: bool,
    zone: usize,
) -> usize {
    debug_assert!(zone < N_ZONES_ROI);
    let r = usize::from(role) - 1;
    let camp = if perspective_blanche { Color::White } else { Color::Black };
    let plan = if couleur == camp { r } else { 6 + r };
    let case_vue = if perspective_blanche {
        usize::from(case)
    } else {
        usize::from(case) ^ 56
    };
    zone * 768 + plan * 64 + case_vue
}

/// Énumère les retraits (signe -1) puis ajouts (signe +1) de pièces induits
/// par `m` joué par `nous`, dans l'ordre exact des deltas de `pousse`, et
/// appelle `delta(couleur, role, case, signe)` pour chacun. Partagé par les
/// deltas du schéma roi-zones (les deux perspectives, ou une seule quand
/// l'autre est reconstruite).
pub(crate) fn pour_chaque_delta(
    nous: Color,
    m: &Move,
    mut delta: impl FnMut(Color, Role, Square, f32),
) {
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
            // Convention shakmaty : `Move::to()` est la case de la TOUR ; les
            // cases d'arrivée réelles viennent du côté de roque.
            let cote = m.castling_side().expect("Move::Castle a toujours un côté");
            delta(nous, Role::King, *king, -1.0);
            delta(nous, Role::Rook, *rook, -1.0);
            delta(nous, Role::King, cote.king_to(nous), 1.0);
            delta(nous, Role::Rook, cote.rook_to(nous), 1.0);
        }
        Move::Put { .. } => unreachable!("Move::Put n'existe qu'en Crazyhouse"),
    }
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
    /// En schéma roi-zones, un coup de roi qui change de zone déclenche la
    /// RECONSTRUCTION de l'accumulateur de sa perspective (voir `pousse_roi8`).
    pub fn pousse(&mut self, eval: &EvalIncrementale, pos_avant: &Chess, m: &Move) {
        debug_assert_eq!(self.h1, eval.h1, "pousse: EvalIncrementale d'une autre taille");
        if eval.schema == SchemaFeatures::RoiZones8 {
            return self.pousse_roi8(eval, pos_avant, m);
        }
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

    /// `pousse` du schéma roi-zones. Tant que le roi d'une perspective reste
    /// dans sa zone, cette perspective reçoit les mêmes deltas de colonnes
    /// que le schéma classique (indexés par SA zone). Si le coup fait CHANGER
    /// DE ZONE le roi du camp qui joue (coup de roi ordinaire — capture
    /// comprise — ou roque, typiquement le grand roque e1→c1), toutes les
    /// features de pièces de SA perspective changent de bloc de 768 :
    /// l'accumulateur de cette perspective est RECONSTRUIT en entier depuis la
    /// position après le coup (le standard NNUE), l'autre perspective gardant
    /// ses deltas — le roi adverse n'ayant pas bougé, sa zone est inchangée.
    fn pousse_roi8(&mut self, eval: &EvalIncrementale, pos_avant: &Chess, m: &Move) {
        let h1 = self.h1;
        let base = self.base_sommet();
        // Duplique le sommet : le nouvel étage part de la position courante.
        self.donnees.extend_from_within(base..);
        let sommet = self.donnees.len() - 2 * h1;

        let nous = pos_avant.turn();
        let nous_blanc = nous == Color::White;
        let (zone_blanche, zone_noire) = zones_rois(pos_avant);

        // Case d'arrivée de NOTRE roi s'il bouge (coup ordinaire ou roque —
        // convention shakmaty : la case d'arrivée d'un Move::Castle est celle
        // de la TOUR, celle du roi vient du côté de roque).
        let arrivee_roi = match m {
            Move::Normal { role: Role::King, to, .. } => Some(*to),
            Move::Castle { .. } => {
                let cote = m.castling_side().expect("Move::Castle a toujours un côté");
                Some(cote.king_to(nous))
            }
            _ => None,
        };
        // Nouvelle zone de NOTRE perspective, seulement si son roi en change.
        let zone_apres = arrivee_roi.and_then(|case| {
            let (avant, apres) = if nous_blanc {
                (zone_blanche, zone_roi(usize::from(case)))
            } else {
                (zone_noire, zone_roi(usize::from(case) ^ 56))
            };
            (apres != avant).then_some(apres)
        });

        let (blanc, noir) = self.donnees[sommet..].split_at_mut(h1);
        match zone_apres {
            // Aucune frontière traversée : deltas dans les DEUX perspectives,
            // chacune indexée par la zone (inchangée) de SON roi.
            None => pour_chaque_delta(nous, m, |couleur, role, case, signe| {
                let ib = indice_piece_roi8(couleur, role, case, true, zone_blanche);
                let inoir = indice_piece_roi8(couleur, role, case, false, zone_noire);
                accumule(blanc, eval.colonne(ib), signe);
                accumule(noir, eval.colonne(inoir), signe);
            }),
            // NOTRE roi change de zone : reconstruction complète de NOTRE
            // perspective, deltas ordinaires pour l'autre.
            Some(zone) => {
                // `pousse` ne reçoit pas la position d'arrivée : on la
                // rejoue. `play_unchecked` suffit, la pile ne reçoit que des
                // coups légaux (contrat de la recherche).
                let mut apres = pos_avant.clone();
                apres.play_unchecked(m);

                let (reconstruit, garde, zone_gardee) = if nous_blanc {
                    (blanc, noir, zone_noire)
                } else {
                    (noir, blanc, zone_blanche)
                };
                reconstruit.copy_from_slice(&eval.biais1);
                for (case, piece) in apres.board().iter() {
                    let idx =
                        indice_piece_roi8(piece.color, piece.role, case, nous_blanc, zone);
                    accumule(reconstruit, eval.colonne(idx), 1.0);
                }
                pour_chaque_delta(nous, m, |couleur, role, case, signe| {
                    let idx = indice_piece_roi8(couleur, role, case, !nous_blanc, zone_gardee);
                    accumule(garde, eval.colonne(idx), signe);
                });
            }
        }
    }

    /// Null-move : la position est inchangée, seul le trait s'inverse — les
    /// accumulateurs sont donc identiques (l'évaluation lira simplement
    /// l'autre perspective), dans les DEUX schémas : les zones roi-zones
    /// dépendent des cases des rois, pas du trait. On duplique le sommet pour
    /// garder la symétrie pousse/depousse de la recherche.
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
    /// (seul l'ordre des sommations f32 de la couche 1 diffère) ; en schéma
    /// roi-zones, égal de même à `forward_actifs`/`evalue_position`.
    pub fn evalue(&self, eval: &EvalIncrementale, pos: &Chess) -> f32 {
        debug_assert_eq!(self.h1, eval.h1, "evalue: EvalIncrementale d'une autre taille");
        let h1 = self.h1;
        let base = self.base_sommet();
        let nous = pos.turn();
        let sommet = &self.donnees[base..];
        let accu = if nous == Color::White { &sommet[..h1] } else { &sommet[h1..] };

        // Copie de travail : le sommet de pile ne doit pas être modifié.
        let mut courant: Vec<f32> = accu.to_vec();

        // Drapeaux non incrémentaux, mêmes conditions que `features::encode`
        // et `features_roi::actifs` : notre O-O, notre O-O-O, leur O-O, leur
        // O-O-O, en passant légal. Seule la BASE des colonnes dépend du
        // schéma (768 en classique, 6144 en roi-zones).
        let base_drapeaux = match eval.schema {
            SchemaFeatures::Classique773 => BASE_DRAPEAUX,
            SchemaFeatures::RoiZones8 => BASE_DRAPEAUX_ROI8,
        };
        let eux = nous.other();
        let roques = pos.castles();
        if roques.has(nous, CastlingSide::KingSide) {
            accumule(&mut courant, eval.colonne(base_drapeaux), 1.0);
        }
        if roques.has(nous, CastlingSide::QueenSide) {
            accumule(&mut courant, eval.colonne(base_drapeaux + 1), 1.0);
        }
        if roques.has(eux, CastlingSide::KingSide) {
            accumule(&mut courant, eval.colonne(base_drapeaux + 2), 1.0);
        }
        if roques.has(eux, CastlingSide::QueenSide) {
            accumule(&mut courant, eval.colonne(base_drapeaux + 3), 1.0);
        }
        if pos.ep_square(EnPassantMode::Legal).is_some() {
            accumule(&mut courant, eval.colonne(base_drapeaux + 4), 1.0);
        }

        // ReLU de la couche 1 (les pré-activations deviennent des activations).
        for v in courant.iter_mut() {
            *v = v.max(0.0);
        }

        // Couches supérieures : boucle GÉNÉRIQUE sur `sup` (une, deux ou
        // davantage de couches), mêmes boucles que `Mlp::avancer` (row-major,
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
    use crate::nn::evalue_position;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::{Board, CastlingMode, FromSetup, Piece, Setup};

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
        let pas_colonnes = vec![0u64; sizes[0]];
        Mlp {
            sizes,
            weights,
            biases,
            adam_mw: zw.clone(),
            adam_vw: zw,
            adam_mb: zb.clone(),
            adam_vb: zb,
            steps: 0,
            pas_colonnes,
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

    /// Garde de EvalIncrementale::new : un réseau à sortie non scalaire est
    /// refusé à la construction (`evalue` lirait `courant[0]` en ignorant
    /// silencieusement les autres sorties).
    #[test]
    #[should_panic(expected = "sortie scalaire")]
    fn eval_incrementale_refuse_sortie_non_scalaire() {
        // Mlp construit par les champs publics : sizes se termine par 2.
        let sizes = vec![N_FEATURES, 4, 2];
        let weights = vec![vec![0.0f32; 4 * N_FEATURES], vec![0.0f32; 2 * 4]];
        let biases = vec![vec![0.0f32; 4], vec![0.0f32; 2]];
        let zw: Vec<Vec<f32>> = weights.iter().map(|w| vec![0.0; w.len()]).collect();
        let zb: Vec<Vec<f32>> = biases.iter().map(|b| vec![0.0; b.len()]).collect();
        let net = Mlp {
            pas_colonnes: vec![0u64; sizes[0]],
            sizes,
            weights,
            biases,
            adam_mw: zw.clone(),
            adam_vw: zw,
            adam_mb: zb.clone(),
            adam_vb: zb,
            steps: 0,
        };
        let _ = EvalIncrementale::new(&net);
    }

    // -----------------------------------------------------------------------
    // 6. Mêmes garanties de parité pour des architectures NON par défaut,
    // créées par `Mlp::new_avec_tailles` (graine fixe) : le réseau ÉLARGI
    // [773,1024,128,1] (cible de la distillation) et une tête PROFONDE
    // [773,256,64,32,1] (trois couches au-dessus de l'accumulateur).
    // -----------------------------------------------------------------------

    /// Batterie complète pour une architecture donnée, reprenant les mêmes
    /// vérifications que les tests 1 à 4 : parité statique, parités
    /// incrémentales sur parties aléatoires ET scriptées (les quatre roques,
    /// promotions avec/sans capture, prises en passant des deux camps),
    /// null-move (avec retour exact après depousse) et marche pousse/depousse.
    /// Les effectifs sont réduits par rapport aux tests 1-2 : les forwards de
    /// référence de ces réseaux larges coûtent cher en debug.
    fn batterie_parite(
        tailles: &[usize],
        graine: u64,
        n_statique: usize,
        n_parties: u64,
        contexte: &str,
    ) {
        // Réseau créé par le constructeur PUBLIC générique ; biais rendus non
        // nuls (comme au test 1) pour qu'une perte des biais soit détectée.
        let mut net = Mlp::new_avec_tailles(tailles, graine);
        assert_eq!(net.sizes, tailles);
        let mut rng = StdRng::seed_from_u64(graine ^ 0xB1A15);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        let eval = EvalIncrementale::new(&net);

        // --- Parité statique sur des positions de parties aléatoires. ---
        let mut positions = vec![Chess::default()];
        let mut g = 0u64;
        while positions.len() < n_statique {
            for (pos, _) in partie_aleatoire(&Chess::default(), graine * 1000 + g, 90) {
                positions.push(pos);
                if positions.len() >= n_statique {
                    break;
                }
            }
            g += 1;
        }
        for (i, pos) in positions.iter().enumerate() {
            let attendu = reference(&net, pos);
            let obtenu = eval.racine(pos).evalue(&eval, pos);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "{contexte}, statique {i} : incrémental {obtenu} vs référence {attendu}"
            );
        }

        // --- Parité incrémentale : parties aléatoires + scripts des tests
        // 2b/2c/2c bis/2d, qui GARANTISSENT la couverture roques/promotions/
        // en passant indépendamment de l'aléa. ---
        let mut parties: Vec<(Vec<(Chess, Move)>, String)> = Vec::new();
        for p in 0..n_parties {
            parties.push((
                partie_aleatoire(&Chess::default(), graine * 1000 + 500 + p, 120),
                format!("{contexte}, partie aléatoire {p}"),
            ));
        }
        parties.push((
            construit_partie(&Chess::default(), &[
                "e2e4", "d7d5", "g1f3", "b8c6", "f1c4", "c8f5", "e1g1", "d8d6", "d2d3", "e8c8",
            ]),
            format!("{contexte}, O-O blanc / O-O-O noir"),
        ));
        parties.push((
            construit_partie(&Chess::default(), &[
                "d2d4", "e7e5", "c1e3", "f8e7", "b1c3", "g8f6", "d1d2", "e8g8", "e1c1",
            ]),
            format!("{contexte}, O-O-O blanc / O-O noir"),
        ));
        parties.push((
            construit_partie(
                &pos_de_fen("rnbqkb1r/ppppppPp/8/8/8/8/PPPPPPpP/RNBQKB1R w KQkq - 0 1"),
                &["g7h8q", "g2h1n"],
            ),
            format!("{contexte}, promotions avec capture"),
        ));
        parties.push((
            construit_partie(&pos_de_fen("8/4k1P1/8/8/8/8/6p1/4K3 w - - 0 1"), &["g7g8q", "g2g1n"]),
            format!("{contexte}, promotions calmes"),
        ));
        parties.push((
            construit_partie(&Chess::default(), &[
                "e2e4", "g8f6", "e4e5", "d7d5", "e5d6", "c7d6", "g1f3", "b7b5", "f3g1", "b5b4",
                "c2c4", "b4c3",
            ]),
            format!("{contexte}, prises en passant"),
        ));
        let (mut roques, mut promotions, mut en_passants) = (0usize, 0usize, 0usize);
        for (partie, nom) in &parties {
            for (_, m) in partie {
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
            verifie_partie(&net, &eval, partie, nom);
        }
        // Plancher garanti par les seuls scripts (4 roques, 4 promotions, 2 e.p.).
        assert!(
            roques >= 4 && promotions >= 4 && en_passants >= 2,
            "{contexte} : couverture insuffisante ({roques} roques, {promotions} promotions, {en_passants} e.p.)"
        );

        // --- Null-move en cours de partie, et retour EXACT après depousse. ---
        let mut testes = 0;
        for p in 0..3u64 {
            let partie = partie_aleatoire(&Chess::default(), graine * 1000 + 800 + p, 60);
            if partie.is_empty() {
                continue;
            }
            let mut pile = eval.racine(&partie[0].0);
            for (avant, m) in &partie {
                pile.pousse(&eval, avant, m);
                let apres = avant.clone().play(m).expect("coup légal");
                if let Some(inverse) = inverse_trait(&apres) {
                    let avant_null = pile.evalue(&eval, &apres);
                    pile.pousse_null();
                    let obtenu = pile.evalue(&eval, &inverse);
                    let attendu = reference(&net, &inverse);
                    assert!(
                        (attendu - obtenu).abs() <= TOL,
                        "{contexte}, null-move partie {p} : {obtenu} vs {attendu}"
                    );
                    pile.depousse();
                    assert_eq!(
                        pile.evalue(&eval, &apres),
                        avant_null,
                        "{contexte} : depousse du null-move ne restitue pas l'évaluation"
                    );
                    testes += 1;
                }
            }
        }
        assert!(testes > 10, "{contexte} : trop peu de null-moves testés ({testes})");

        // --- Marche aléatoire pousse/depousse, parité après CHAQUE pas. ---
        let mut rng = StdRng::seed_from_u64(graine.wrapping_mul(31) + 7);
        let mut pile = eval.racine(&Chess::default());
        let mut positions = vec![Chess::default()];
        for pas in 0..250 {
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
                break;
            }
            let pos = positions.last().unwrap();
            let attendu = reference(&net, pos);
            let obtenu = pile.evalue(&eval, pos);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "{contexte}, marche pas {pas} (profondeur {}) : {obtenu} vs {attendu}",
                positions.len()
            );
        }
    }

    /// 6a. Réseau ÉLARGI [773,1024,128,1] (architecture cible de la
    /// distillation) : toute la batterie de parité.
    #[test]
    fn parite_reseau_elargi_1024_128() {
        batterie_parite(&[N_FEATURES, 1024, 128, 1], 101, 60, 4, "réseau [773,1024,128,1]");
    }

    /// 6b. Tête PROFONDE [773,256,64,32,1] : trois couches au-dessus de
    /// l'accumulateur, la boucle générique de `evalue` est réellement exercée
    /// au-delà du schéma à deux couches.
    #[test]
    fn parite_tete_profonde_256_64_32() {
        batterie_parite(&[N_FEATURES, 256, 64, 32, 1], 202, 120, 8, "réseau [773,256,64,32,1]");
    }

    // -----------------------------------------------------------------------
    // 7. Schéma ROI-ZONES (RoiZones8) : batterie de parité complète contre la
    // référence CREUSE `evalue_position` (features_roi::actifs +
    // forward_actifs). Les coups de roi qui changent de zone — reconstruction
    // de l'accumulateur d'UNE perspective, l'autre gardant ses deltas — sont
    // LE cas critique : ils sont comptés dans les parties aléatoires et
    // FORCÉS par des marches de roi scriptées.
    // -----------------------------------------------------------------------

    /// Référence du schéma roi-zones : le chemin creux de production
    /// (`evalue_position` route vers `features_roi::actifs` puis
    /// `Mlp::forward_actifs`).
    fn reference_roi8(net: &Mlp, pos: &Chess) -> f32 {
        assert_eq!(net.schema(), SchemaFeatures::RoiZones8);
        let mut tampon = Vec::new();
        evalue_position(net, pos, &mut tampon)
    }

    /// Petit réseau RoiZones8 [6149, h1, h2, 1] créé par le constructeur
    /// PUBLIC `new_roi_zones`, biais rendus NON NULS (mêmes raisons que
    /// `petit_reseau` : une perte des biais doit être détectée).
    fn petit_reseau_roi8(graine: u64, h1: usize, h2: usize) -> Mlp {
        let mut net = Mlp::new_roi_zones(&[N_FEATURES_ROI, h1, h2, 1], graine);
        let mut rng = StdRng::seed_from_u64(graine ^ 0x0A11E5);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        net
    }

    /// Zone du roi de `camp` dans SA perspective (miroir pour les noirs).
    fn zone_du_camp(pos: &Chess, camp: Color) -> usize {
        let roi = pos.board().king_of(camp).expect("roi présent");
        let case = if camp == Color::White {
            usize::from(roi)
        } else {
            usize::from(roi) ^ 56
        };
        zone_roi(case)
    }

    /// Nombre de coups de `partie` où le roi du camp QUI JOUE change de zone
    /// dans SA perspective — chacun déclenche une reconstruction.
    fn nb_changements_zone(partie: &[(Chess, Move)]) -> usize {
        partie
            .iter()
            .filter(|(avant, m)| {
                let nous = avant.turn();
                let apres = avant.clone().play(m).expect("coup légal");
                zone_du_camp(avant, nous) != zone_du_camp(&apres, nous)
            })
            .count()
    }

    /// Rejoue `partie` en maintenant la pile : parité roi-zones exigée APRÈS
    /// CHAQUE coup (pendant de `verifie_partie` pour le schéma creux).
    fn verifie_partie_roi8(
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
            let attendu = reference_roi8(net, &apres);
            let obtenu = pile.evalue(eval, &apres);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "{contexte}, coup {i} ({m:?}) : incrémental {obtenu} vs référence {attendu}"
            );
        }
    }

    /// 7a. Parité statique roi-zones : les 64 COMBINAISONS de zones des deux
    /// rois (positions à deux rois construites — une case par zone et par
    /// perspective —, les deux traits), puis 200 positions de parties
    /// aléatoires (droits de roque et en passant réels).
    #[test]
    fn parite_statique_roi8_toutes_les_zones() {
        let net = petit_reseau_roi8(71, 32, 12);
        let eval = EvalIncrementale::new(&net);

        // Une case par zone, pour chaque perspective (la liste noire est le
        // miroir vertical de la blanche : mêmes zones après `case ^ 56`).
        let cases_blanches = [
            Square::A1, Square::B4, Square::G1, Square::F3,
            Square::C5, Square::D7, Square::E6, Square::H8,
        ];
        let cases_noires = [
            Square::A8, Square::B5, Square::G8, Square::F6,
            Square::C4, Square::D2, Square::E3, Square::H1,
        ];
        for (z, (&cb, &cn)) in cases_blanches.iter().zip(&cases_noires).enumerate() {
            assert_eq!(zone_roi(usize::from(cb)), z, "liste blanche mal ordonnée");
            assert_eq!(zone_roi(usize::from(cn) ^ 56), z, "liste noire mal ordonnée");
        }

        let mut testes = 0;
        for &roi_blanc in &cases_blanches {
            for &roi_noir in &cases_noires {
                // Rois adjacents : position illégale, combinaison sautée.
                let (a, b) = (usize::from(roi_blanc), usize::from(roi_noir));
                let (df, dr) = (
                    ((a & 7) as i32 - (b & 7) as i32).abs(),
                    ((a >> 3) as i32 - (b >> 3) as i32).abs(),
                );
                if df <= 1 && dr <= 1 {
                    continue;
                }
                let mut plateau = Board::empty();
                plateau.set_piece_at(roi_blanc, Piece { color: Color::White, role: Role::King });
                plateau.set_piece_at(roi_noir, Piece { color: Color::Black, role: Role::King });
                for trait_blanc in [true, false] {
                    let mut setup = Setup::empty();
                    setup.board = plateau.clone();
                    setup.turn = if trait_blanc { Color::White } else { Color::Black };
                    let pos = Chess::from_setup(setup, CastlingMode::Standard)
                        .expect("deux rois non adjacents : position légale");
                    let attendu = reference_roi8(&net, &pos);
                    let obtenu = eval.racine(&pos).evalue(&eval, &pos);
                    assert!(
                        (attendu - obtenu).abs() <= TOL,
                        "rois {roi_blanc}/{roi_noir}, trait blanc {trait_blanc} : {obtenu} vs {attendu}"
                    );
                    testes += 1;
                }
            }
        }
        // 64 combinaisons moins les paires adjacentes, deux traits chacune.
        assert!(testes >= 100, "trop peu de combinaisons de zones testées ({testes})");

        // Positions réelles issues de parties aléatoires (roques, e.p.).
        let mut positions = vec![Chess::default()];
        let mut graine = 0u64;
        while positions.len() < 200 {
            for (p, _) in partie_aleatoire(&Chess::default(), 71_000 + graine, 90) {
                positions.push(p);
                if positions.len() >= 200 {
                    break;
                }
            }
            graine += 1;
        }
        for (i, pos) in positions.iter().enumerate() {
            let attendu = reference_roi8(&net, pos);
            let obtenu = eval.racine(pos).evalue(&eval, pos);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "position {i} : incrémental {obtenu} vs référence {attendu}"
            );
        }
    }

    /// 7b. Parité incrémentale roi-zones : 100 parties aléatoires rejouées
    /// coup à coup contre la référence creuse. La couverture (roques,
    /// promotions, prises en passant, coups de roi changeant de zone —
    /// c'est-à-dire reconstructions) est comptée et exigée.
    #[test]
    fn parite_incrementale_roi8_100_parties() {
        let net = petit_reseau_roi8(72, 32, 12);
        let eval = EvalIncrementale::new(&net);
        let (mut roques, mut promotions, mut en_passants) = (0usize, 0usize, 0usize);
        let mut reconstructions = 0usize;
        for g in 0..100u64 {
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
            reconstructions += nb_changements_zone(&partie);
            verifie_partie_roi8(&net, &eval, &partie, &format!("partie aléatoire roi8 {g}"));
        }
        println!(
            "couverture roi8 : {roques} roques, {promotions} promotions, {en_passants} e.p., \
             {reconstructions} changements de zone"
        );
        assert!(roques > 0, "aucun roque rencontré dans les parties aléatoires");
        assert!(promotions > 0, "aucune promotion rencontrée dans les parties aléatoires");
        assert!(en_passants > 0, "aucune prise en passant rencontrée");
        assert!(
            reconstructions > 50,
            "trop peu de changements de zone ({reconstructions}) : le cas critique n'est pas exercé"
        );
    }

    /// 7c. Parties scriptées sous roi-zones : les quatre roques — les DEUX
    /// grands roques font CHANGER le roi de zone (e1→c1 et e8→c8 : zone 2 →
    /// zone 0) quand les petits ne la changent pas, les chemins delta ET
    /// reconstruction du roque sont donc exercés —, les promotions avec et
    /// sans capture, les prises en passant des deux camps.
    #[test]
    fn parite_roi8_scripts_roques_promotions_en_passant() {
        let net = petit_reseau_roi8(73, 24, 8);
        let eval = EvalIncrementale::new(&net);
        let parties: [(Vec<(Chess, Move)>, &str, usize); 5] = [
            (
                construit_partie(&Chess::default(), &[
                    "e2e4", "d7d5", "g1f3", "b8c6", "f1c4", "c8f5", "e1g1", "d8d6", "d2d3", "e8c8",
                ]),
                "O-O blanc / O-O-O noir",
                1, // seul e8c8 traverse une frontière
            ),
            (
                construit_partie(&Chess::default(), &[
                    "d2d4", "e7e5", "c1e3", "f8e7", "b1c3", "g8f6", "d1d2", "e8g8", "e1c1",
                ]),
                "O-O-O blanc / O-O noir",
                1, // seul e1c1 traverse une frontière
            ),
            (
                construit_partie(
                    &pos_de_fen("rnbqkb1r/ppppppPp/8/8/8/8/PPPPPPpP/RNBQKB1R w KQkq - 0 1"),
                    &["g7h8q", "g2h1n"],
                ),
                "promotions avec capture",
                0,
            ),
            (
                construit_partie(
                    &pos_de_fen("8/4k1P1/8/8/8/8/6p1/4K3 w - - 0 1"),
                    &["g7g8q", "g2g1n"],
                ),
                "promotions calmes",
                0,
            ),
            (
                construit_partie(&Chess::default(), &[
                    "e2e4", "g8f6", "e4e5", "d7d5", "e5d6", "c7d6", "g1f3", "b7b5", "f3g1",
                    "b5b4", "c2c4", "b4c3",
                ]),
                "prises en passant",
                0,
            ),
        ];
        for (partie, contexte, changements_attendus) in &parties {
            assert_eq!(
                nb_changements_zone(partie),
                *changements_attendus,
                "{contexte} : compte de changements de zone inattendu"
            );
            verifie_partie_roi8(&net, &eval, partie, contexte);
        }
    }

    /// 7d. LE cas critique, forcé : marches de roi scriptées à travers les
    /// frontières de zones. D'abord les deux rois seuls (12 traversées, dont
    /// des retours en arrière et des coups de roi SANS changement de zone) ;
    /// puis une marche où le roi PREND une pièce en changeant de zone (la
    /// perspective du roi est reconstruite, l'autre retire la victime par
    /// delta).
    #[test]
    fn parite_roi8_marches_de_roi_changements_de_zone() {
        let net = petit_reseau_roi8(74, 24, 8);
        let eval = EvalIncrementale::new(&net);

        // Blanc : a1→b7 puis redescente par les files c-e (zones
        // 0→1→4→5→4→6→3→1→0) ; noir : h8→a1 en écho (2→3→6→7→5).
        let marche = construit_partie(&pos_de_fen("7k/8/8/8/8/8/8/K7 w - - 0 1"), &[
            "a1a2", "h8h7", "a2a3", "h7h6", "a3a4", "h6h5", "a4a5", "h5h4",
            "a5b5", "h4g4", "b5b6", "g4g3", "b6b7", "g3g2", "b7c7", "g2g1",
            "c7c6", "g1f1", "c6d6", "f1e1", "d6e6", "e1d1", "e6e5", "d1c1",
            "e5e4", "c1b1", "e4d4", "b1b2", "d4d3", "b2a2", "d3d2", "a2a1",
        ]);
        assert_eq!(
            nb_changements_zone(&marche),
            12,
            "la marche des deux rois doit traverser 12 frontières"
        );
        verifie_partie_roi8(&net, &eval, &marche, "marche des deux rois");

        // Kxa7 : capture PAR le roi en traversant une frontière (zone 4 → 5),
        // puis chaque roi redescend en re-traversant plusieurs frontières.
        let capture = construit_partie(&pos_de_fen("8/b4k2/K7/8/8/8/1P6/8 w - - 0 1"), &[
            "a6a7", "f7e6", "a7b6", "e6d5", "b6b5", "d5e4", "b5b4", "e4f3",
        ]);
        assert!(
            matches!(&capture[0].1, Move::Normal { role: Role::King, capture: Some(_), .. }),
            "le premier coup doit être une capture par le roi"
        );
        assert_eq!(
            nb_changements_zone(&capture),
            6,
            "compte de changements de zone de la marche avec capture"
        );
        verifie_partie_roi8(&net, &eval, &capture, "marche avec capture par le roi");
    }

    /// 7e. Null-move roi-zones : les accumulateurs sont inchangés (les zones
    /// dépendent des rois, pas du trait), l'évaluation lit simplement l'autre
    /// perspective. Parité sur des plateaux fixes — dont des rois dans des
    /// zones DIFFÉRENTES — puis en cours de parties aléatoires, avec retour
    /// EXACT après depousse.
    #[test]
    fn pousse_null_roi8_echange_les_perspectives() {
        let net = petit_reseau_roi8(75, 24, 8);
        let eval = EvalIncrementale::new(&net);

        // Paires (trait blanc, trait noir) sur le même plateau, sans e.p. :
        // départ, position roquable, rois excentrés en zones 2/0 puis 3/3.
        let paires = [
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
            ),
            (
                "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
                "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R b KQkq - 0 1",
            ),
            ("8/2k5/8/8/8/8/5PPP/6K1 w - - 0 1", "8/2k5/8/8/8/8/5PPP/6K1 b - - 0 1"),
            ("8/5p2/4k3/8/8/4K3/5P2/8 w - - 0 1", "8/5p2/4k3/8/8/4K3/5P2/8 b - - 0 1"),
        ];
        for (fen_blanc, fen_noir) in paires {
            let pos = pos_de_fen(fen_blanc);
            let pos_inverse = pos_de_fen(fen_noir);

            let mut pile = eval.racine(&pos);
            let origine = pile.evalue(&eval, &pos);
            pile.pousse_null();
            let obtenu = pile.evalue(&eval, &pos_inverse);
            let attendu = reference_roi8(&net, &pos_inverse);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "null-move roi8 sur {fen_blanc} : {obtenu} vs {attendu}"
            );
            // Dépiler le null-move restitue exactement l'évaluation d'origine.
            pile.depousse();
            assert_eq!(pile.evalue(&eval, &pos), origine);
        }

        // Null-move en cours de partie, et retour EXACT après depousse.
        let mut testes = 0;
        for g in 0..6u64 {
            let partie = partie_aleatoire(&Chess::default(), 7500 + g, 60);
            if partie.is_empty() {
                continue;
            }
            let mut pile = eval.racine(&partie[0].0);
            for (avant, m) in &partie {
                pile.pousse(&eval, avant, m);
                let apres = avant.clone().play(m).expect("coup légal");
                if let Some(inverse) = inverse_trait(&apres) {
                    let avant_null = pile.evalue(&eval, &apres);
                    pile.pousse_null();
                    let obtenu = pile.evalue(&eval, &inverse);
                    let attendu = reference_roi8(&net, &inverse);
                    assert!(
                        (attendu - obtenu).abs() <= TOL,
                        "null-move roi8 en partie {g} : {obtenu} vs {attendu}"
                    );
                    pile.depousse();
                    assert_eq!(
                        pile.evalue(&eval, &apres),
                        avant_null,
                        "depousse du null-move roi8 ne restitue pas l'évaluation"
                    );
                    testes += 1;
                }
            }
        }
        assert!(testes > 20, "trop peu de null-moves roi8 testés ({testes})");
    }

    /// 7f. Marche aléatoire pousse/depousse (60 % / 40 %) sur une finale rois
    /// et pions — la plupart des coups sont des coups de roi : des
    /// reconstructions de zone sont poussées PUIS dépilées en permanence
    /// (celles dépilées doivent disparaître sans trace). Parité avec la
    /// référence creuse après CHAQUE pas.
    #[test]
    fn depousse_roi8_rejoint_la_reference() {
        let net = petit_reseau_roi8(76, 24, 8);
        let eval = EvalIncrementale::new(&net);
        let mut rng = StdRng::seed_from_u64(770);

        let depart = pos_de_fen("7k/5p2/8/8/8/8/2P5/K7 w - - 0 1");
        let mut pile = eval.racine(&depart);
        let mut positions = vec![depart];
        let mut changements = 0usize;
        for pas in 0..900 {
            let sommet = positions.last().unwrap().clone();
            let coups = sommet.legal_moves();
            let pousser = !coups.is_empty() && (positions.len() == 1 || rng.gen_bool(0.6));
            if pousser {
                let m = coups[rng.gen_range(0..coups.len())].clone();
                let nous = sommet.turn();
                let zone_avant = zone_du_camp(&sommet, nous);
                pile.pousse(&eval, &sommet, &m);
                let apres = sommet.play(&m).expect("coup légal");
                if zone_du_camp(&apres, nous) != zone_avant {
                    changements += 1;
                }
                positions.push(apres);
            } else if positions.len() > 1 {
                pile.depousse();
                positions.pop();
            } else {
                break; // partie terminée à la racine
            }
            let pos = positions.last().unwrap();
            let attendu = reference_roi8(&net, pos);
            let obtenu = pile.evalue(&eval, pos);
            assert!(
                (attendu - obtenu).abs() <= TOL,
                "pas {pas} (profondeur {}) : {obtenu} vs {attendu}",
                positions.len()
            );
        }
        assert!(
            changements > 30,
            "trop peu de changements de zone poussés dans la marche ({changements})"
        );
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

    // -----------------------------------------------------------------------
    // 8. STRESS DE PARITÉ EXHAUSTIF roi8 (chantier forensique) — tests
    // #[ignore], pensés pour le release. La pile est pilotée DIRECTEMENT sur
    // des séquences profondes ; à CHAQUE pli (après pousse, sous pousse_null
    // ET après chaque depousse) l'évaluation incrémentale est comparée à la
    // référence creuse recalculée de zéro. Deux réseaux systématiquement :
    // un roi8 aléatoire aux tailles réelles [6149,1024,128,1] et le réseau
    // RÉEL models/chess_latest.bin (s'il est présent et au schéma roi8).
    //
    // Lancement complet (release, quelques minutes) :
    //   cargo test --release --lib nnue::tests::stress_roi8 -- --ignored --nocapture
    // Fumé (debug, volume réduit) — PowerShell :
    //   $env:NNUE_STRESS_PARTIES='2'
    //   cargo test --lib nnue::tests::stress_roi8 -- --ignored --nocapture
    // -----------------------------------------------------------------------

    /// Tolérance du stress. La sortie est un tanh dans (-1,1) : une tolérance
    /// purement RELATIVE n'a pas de sens près de 0, on borne donc l'écart
    /// ABSOLU. 1e-3 équivaut à ~1e-4 relatif ramené à la dynamique utile de
    /// la sortie (|v| jusqu'à ~1) — c'est le critère « 1e-4 relatif » du
    /// cahier des charges exprimé en absolu. Le seul écart légitime est la
    /// dérive f32 des accumulateurs (ordre des sommations différent + des
    /// centaines de deltas cumulés sur >120 plis), attendue sous ~2e-4 sur
    /// [6149,1024,128,1] ; un bug d'indexation déplace au moins une colonne
    /// entière et produit des écarts d'ordre 1e-2 à 1. 1e-3 sépare les deux
    /// populations avec une marge d'au moins ×5 de chaque côté.
    const TOL_STRESS: f32 = 1e-3;

    /// Volume du stress : nombre de parties aléatoires (défaut 2000), réglable
    /// par la variable d'environnement NNUE_STRESS_PARTIES pour un fumé rapide
    /// en debug. Les autres tests de stress se mettent à l'échelle.
    fn nb_parties_stress() -> u64 {
        std::env::var("NNUE_STRESS_PARTIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2000)
    }

    /// Réseaux soumis au stress : un roi8 aléatoire AUX TAILLES RÉELLES
    /// [6149,1024,128,1] (biais rendus non nuls), puis le réseau réel
    /// models/chess_latest.bin s'il se charge ET est au schéma roi8 (train.exe
    /// peut être en train de le réécrire : un chargement raté est signalé
    /// et toléré, le réseau aléatoire couvre alors seul la logique).
    fn reseaux_stress() -> Vec<(Mlp, String)> {
        let mut nets = vec![(
            petit_reseau_roi8(0xACE, 1024, 128),
            "roi8 aléatoire [6149,1024,128,1]".to_string(),
        )];
        let chemin = concat!(env!("CARGO_MANIFEST_DIR"), "/models/chess_latest.bin");
        match Mlp::load(chemin) {
            Ok(net) if net.schema() == SchemaFeatures::RoiZones8 => {
                println!("STRESS : réseau réel chargé ({chemin}, tailles {:?})", net.sizes);
                nets.push((net, "réseau réel chess_latest.bin".to_string()));
            }
            Ok(net) => println!(
                "STRESS : {chemin} au schéma {:?} (pas roi8), réseau réel non testé",
                net.schema()
            ),
            Err(e) => println!("STRESS : {chemin} illisible ({e}), réseau réel non testé"),
        }
        nets
    }

    /// Bilan d'une séquence de stress : la couverture RÉELLEMENT exercée.
    #[derive(Default)]
    struct StatsStress {
        plis: usize,
        changements_zone: usize,
        roques: usize,
        promotions: usize,
        en_passants: usize,
        nulls_testes: usize,
    }

    impl StatsStress {
        fn cumule(&mut self, autre: &StatsStress) {
            self.plis += autre.plis;
            self.changements_zone += autre.changements_zone;
            self.roques += autre.roques;
            self.promotions += autre.promotions;
            self.en_passants += autre.en_passants;
            self.nulls_testes += autre.nulls_testes;
        }

        fn affiche(&self, contexte: &str) {
            println!(
                "{contexte} : {} plis, {} changements de zone, {} roques, {} promotions, \
                 {} e.p., {} null-moves vérifiés",
                self.plis, self.changements_zone, self.roques, self.promotions,
                self.en_passants, self.nulls_testes
            );
        }
    }

    /// Compare l'évaluation incrémentale du sommet de pile à la référence
    /// creuse recalculée de zéro ; au-delà de TOL_STRESS (ou si l'une des
    /// valeurs est NaN), imprime le rapport complet — FEN de départ, séquence
    /// UCI, pli, valeurs — puis panique. C'est la matière première du
    /// diagnostic : la séquence rejouée telle quelle reproduit la divergence.
    #[allow(clippy::too_many_arguments)]
    fn exige_parite_roi8(
        net: &Mlp,
        eval: &EvalIncrementale,
        pile: &PileAccus,
        pos: &Chess,
        fen_depart: &str,
        ucis: &[String],
        pli: usize,
        etape: &str,
    ) {
        let attendu = reference_roi8(net, pos);
        let obtenu = pile.evalue(eval, pos);
        let ecart = (attendu - obtenu).abs();
        // `!(ecart <= TOL)` plutôt que `ecart > TOL` : attrape aussi les NaN.
        if !(ecart <= TOL_STRESS) {
            let fen_sommet = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
            panic!(
                "\n=== DIVERGENCE NNUE roi8 ===\n\
                 étape        : {etape}\n\
                 pli          : {pli}\n\
                 FEN départ   : {fen_depart}\n\
                 FEN sommet   : {fen_sommet}\n\
                 séquence UCI : {}\n\
                 incrémental  : {obtenu:.8}\n\
                 exact        : {attendu:.8}\n\
                 écart        : {ecart:.3e} (tolérance {TOL_STRESS:.0e})\n",
                if ucis.is_empty() { "(aucun coup)".to_string() } else { ucis.join(" ") }
            );
        }
    }

    /// Cœur du stress : déroule `coups` (légaux, dans l'ordre) depuis `depart`
    /// en pilotant la pile roi8.
    /// - Parité exigée après CHAQUE pousse ;
    /// - un pousse_null intercalé UN COUP SUR HUIT : parité au trait inversé
    ///   (quand la position inversée est légale) puis depousse et parité de
    ///   retour — le null dépilé ne doit laisser aucune trace ;
    /// - à la fin, dépilage INTÉGRAL de la séquence, parité exigée après
    ///   CHAQUE depousse (l'état restauré doit être parfait même quand les
    ///   étages dépilés contenaient des reconstructions de zone).
    fn stress_deroule_roi8(
        net: &Mlp,
        eval: &EvalIncrementale,
        depart: &Chess,
        coups: &[Move],
        contexte: &str,
    ) -> StatsStress {
        let fen_depart = Fen::from_position(depart.clone(), EnPassantMode::Legal).to_string();
        let mut stats = StatsStress::default();
        let mut pile = eval.racine(depart);
        let mut positions = vec![depart.clone()];
        let mut ucis: Vec<String> = Vec::new();
        for (pli, m) in coups.iter().enumerate() {
            let sommet = positions.last().unwrap().clone();
            let nous = sommet.turn();
            let zone_avant = zone_du_camp(&sommet, nous);
            pile.pousse(eval, &sommet, m);
            let apres = sommet.play(m).expect("coup légal");
            ucis.push(m.to_uci(CastlingMode::Standard).to_string());
            stats.plis += 1;
            if zone_du_camp(&apres, nous) != zone_avant {
                stats.changements_zone += 1;
            }
            if m.is_castle() {
                stats.roques += 1;
            }
            if m.promotion().is_some() {
                stats.promotions += 1;
            }
            if m.is_en_passant() {
                stats.en_passants += 1;
            }
            positions.push(apres.clone());
            exige_parite_roi8(
                net, eval, &pile, &apres, &fen_depart, &ucis, ucis.len(),
                &format!("{contexte} — après pousse"),
            );
            // Null-move intercalé un coup sur huit, comme en recherche.
            if pli % 8 == 3 {
                pile.pousse_null();
                if let Some(inverse) = inverse_trait(&apres) {
                    exige_parite_roi8(
                        net, eval, &pile, &inverse, &fen_depart, &ucis, ucis.len(),
                        &format!("{contexte} — sous pousse_null (trait inversé)"),
                    );
                    stats.nulls_testes += 1;
                }
                pile.depousse();
                exige_parite_roi8(
                    net, eval, &pile, &apres, &fen_depart, &ucis, ucis.len(),
                    &format!("{contexte} — après depousse du null-move"),
                );
            }
        }
        // Redescente : dépilage intégral, parité après CHAQUE depousse.
        while positions.len() > 1 {
            pile.depousse();
            positions.pop();
            let pos = positions.last().unwrap();
            exige_parite_roi8(
                net, eval, &pile, pos, &fen_depart, &ucis, positions.len() - 1,
                &format!("{contexte} — redescente, après depousse (retour au pli {})", positions.len() - 1),
            );
        }
        stats
    }

    /// Les coups d'une partie RandomBot depuis `depart` (mêmes conditions
    /// d'arrêt que `partie_aleatoire`, dont elle est un simple habillage).
    fn coups_aleatoires(depart: &Chess, graine: u64, max_plis: usize) -> Vec<Move> {
        partie_aleatoire(depart, graine, max_plis)
            .into_iter()
            .map(|(_, m)| m)
            .collect()
    }

    /// 8a. Des MILLIERS de parties aléatoires longues (graines fixes, jusqu'à
    /// 200 plis) depuis la position initiale, null-move intercalé un coup sur
    /// huit, dépilage intégral vérifié — sur le réseau aléatoire ET le réseau
    /// réel. La couverture naturelle (roques, promotions, prises en passant,
    /// changements de zone) est comptée et exigée au volume nominal.
    #[test]
    #[ignore]
    fn stress_roi8_parties_aleatoires_null_move() {
        let n_parties = nb_parties_stress();
        for (net, nom) in &reseaux_stress() {
            let eval = EvalIncrementale::new(net);
            let mut total = StatsStress::default();
            for g in 0..n_parties {
                let coups = coups_aleatoires(&Chess::default(), 0x57E5_0000 + g, 200);
                total.cumule(&stress_deroule_roi8(
                    net, &eval, &Chess::default(), &coups,
                    &format!("[{nom}] partie aléatoire {g}"),
                ));
            }
            total.affiche(&format!("[{nom}] {n_parties} parties aléatoires"));
            // Planchers de couverture au volume nominal seulement : un fumé
            // à 2 parties ne peut pas garantir une promotion.
            if n_parties >= 100 {
                assert!(total.roques > 0, "[{nom}] aucun roque exercé");
                assert!(total.promotions > 0, "[{nom}] aucune promotion exercée");
                assert!(total.en_passants > 0, "[{nom}] aucune prise en passant exercée");
                assert!(
                    total.changements_zone > n_parties as usize,
                    "[{nom}] trop peu de changements de zone ({})",
                    total.changements_zone
                );
                assert!(total.nulls_testes > 100, "[{nom}] trop peu de null-moves vérifiés");
            }
        }
    }

    /// 8b. Scripts déterministes : les QUATRE roques (les deux grands font
    /// changer le roi de zone, les deux petits non — chemins delta ET
    /// reconstruction), les promotions vers les QUATRE pièces avec puis sans
    /// capture (deux camps à chaque fois), les prises en passant des deux
    /// camps. Chaque script passe par le déroulé complet (null-moves
    /// intercalés, dépilage intégral vérifié).
    #[test]
    #[ignore]
    fn stress_roi8_scripts_roques_promotions_en_passant() {
        // (FEN ou None = position initiale, coups UCI, contexte)
        let mut scripts: Vec<(Option<&str>, Vec<String>, String)> = vec![
            (
                None,
                ["e2e4", "d7d5", "g1f3", "b8c6", "f1c4", "c8f5", "e1g1", "d8d6", "d2d3", "e8c8"]
                    .iter().map(|s| s.to_string()).collect(),
                "O-O blanc / O-O-O noir (e8c8 change de zone)".to_string(),
            ),
            (
                None,
                ["d2d4", "e7e5", "c1e3", "f8e7", "b1c3", "g8f6", "d1d2", "e8g8", "e1c1"]
                    .iter().map(|s| s.to_string()).collect(),
                "O-O-O blanc / O-O noir (e1c1 change de zone)".to_string(),
            ),
            (
                None,
                ["e2e4", "g8f6", "e4e5", "d7d5", "e5d6", "c7d6", "g1f3", "b7b5", "f3g1",
                 "b5b4", "c2c4", "b4c3"]
                    .iter().map(|s| s.to_string()).collect(),
                "prises en passant des deux camps".to_string(),
            ),
        ];
        // Promotions : les 4 pièces, avec capture (g7xh8 / g2xh1) puis
        // calmes (g7g8 / g2g1), un camp puis l'autre à chaque script.
        // Roi noir en b7 : hors de portée des QUATRE promotions blanches en
        // g8 (en e7, la sous-promotion cavalier g8=N donnerait échec et
        // rendrait la réponse noire illégale).
        for piece in ["q", "r", "b", "n"] {
            scripts.push((
                Some("rnbqkb1r/ppppppPp/8/8/8/8/PPPPPPpP/RNBQKB1R w KQkq - 0 1"),
                vec![format!("g7h8{piece}"), format!("g2h1{piece}")],
                format!("promotions {piece} avec capture"),
            ));
            scripts.push((
                Some("8/1k4P1/8/8/8/8/6p1/4K3 w - - 0 1"),
                vec![format!("g7g8{piece}"), format!("g2g1{piece}")],
                format!("promotions {piece} calmes"),
            ));
        }

        for (net, nom) in &reseaux_stress() {
            let eval = EvalIncrementale::new(net);
            let mut total = StatsStress::default();
            for (fen, ucis, contexte) in &scripts {
                let depart = match fen {
                    Some(f) => pos_de_fen(f),
                    None => Chess::default(),
                };
                let refs: Vec<&str> = ucis.iter().map(|s| s.as_str()).collect();
                let coups: Vec<Move> = construit_partie(&depart, &refs)
                    .into_iter()
                    .map(|(_, m)| m)
                    .collect();
                total.cumule(&stress_deroule_roi8(
                    net, &eval, &depart, &coups,
                    &format!("[{nom}] script {contexte}"),
                ));
            }
            total.affiche(&format!("[{nom}] scripts"));
            // Couverture garantie par construction : 4 roques (dont 2 avec
            // changement de zone), 16 promotions, 2 prises en passant.
            assert_eq!(total.roques, 4, "[{nom}] les quatre roques doivent être exercés");
            assert_eq!(total.promotions, 16, "[{nom}] 4 pièces × capture/calme × 2 camps");
            assert_eq!(total.en_passants, 2, "[{nom}] une prise en passant par camp");
            assert!(
                total.changements_zone >= 2,
                "[{nom}] les grands roques doivent changer la zone du roi"
            );
        }
    }

    /// 8c. Finales générées par `departs` (rois + pions et petites finales :
    /// trafic maximal de changements de zone par marches de roi), parties
    /// aléatoires longues avec null-moves et dépilage intégral vérifié.
    #[test]
    #[ignore]
    fn stress_roi8_finales_marches_de_roi() {
        let n_parties = (nb_parties_stress() / 4).max(1);
        // Départs tirés UNE fois (graine fixe) : mêmes finales pour tous les
        // réseaux, comparaisons reproductibles.
        let mut rng = StdRng::seed_from_u64(0xF1A1E5);
        let departs: Vec<(Chess, String)> = (0..n_parties)
            .map(|i| {
                let d = crate::departs::tirage(&mut rng, 0.0, 1.0);
                (d.pos, format!("finale {i} ({})", d.etiquette))
            })
            .collect();
        for (net, nom) in &reseaux_stress() {
            let eval = EvalIncrementale::new(net);
            let mut total = StatsStress::default();
            for (i, (depart, etiquette)) in departs.iter().enumerate() {
                let coups = coups_aleatoires(depart, 0xF00_0000 + i as u64, 240);
                total.cumule(&stress_deroule_roi8(
                    net, &eval, depart, &coups,
                    &format!("[{nom}] {etiquette}"),
                ));
            }
            total.affiche(&format!("[{nom}] {n_parties} finales"));
            if n_parties >= 50 {
                assert!(
                    total.changements_zone > 2 * n_parties as usize,
                    "[{nom}] les finales doivent produire un trafic massif de zones ({})",
                    total.changements_zone
                );
            }
        }
    }

    /// 8d. Montagnes pousse/depousse : marche aléatoire profonde (60 % pousse
    /// / 40 % depousse, coups de roi favorisés à 70 %) sur des finales de rois
    /// et une position roquable — la pile traverse des reconstructions de zone
    /// en montée PUIS les dépile ; parité exigée après CHAQUE pas, montée
    /// comme descente.
    #[test]
    #[ignore]
    fn stress_roi8_montagnes_pousse_depousse() {
        let pas_total = (nb_parties_stress().saturating_mul(4).max(50)) as usize;
        let departs = [
            ("7k/5p2/8/8/8/8/2P5/K7 w - - 0 1", "roi+pion contre roi+pion"),
            ("8/1k6/8/4p3/3P4/8/6K1/8 w - - 0 1", "pions bloqués, rois libres"),
            (
                "r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1",
                "milieu roquable",
            ),
        ];
        for (net, nom) in &reseaux_stress() {
            let eval = EvalIncrementale::new(net);
            for (graine, (fen, etiquette)) in departs.iter().enumerate() {
                let depart = pos_de_fen(fen);
                let mut rng = StdRng::seed_from_u64(0xA10_000 + graine as u64);
                let mut pile = eval.racine(&depart);
                let mut positions = vec![depart.clone()];
                let mut ucis: Vec<String> = Vec::new();
                let mut changements = 0usize;
                let contexte = format!("[{nom}] montagne {etiquette}");
                for pas in 0..pas_total {
                    let sommet = positions.last().unwrap().clone();
                    let coups = sommet.legal_moves();
                    let pousser =
                        !coups.is_empty() && (positions.len() == 1 || rng.gen_bool(0.6));
                    if pousser {
                        // Coups de roi favorisés : trafic maximal de zones.
                        let de_roi: Vec<&Move> =
                            coups.iter().filter(|m| m.role() == Role::King).collect();
                        let m = if !de_roi.is_empty() && rng.gen_bool(0.7) {
                            de_roi[rng.gen_range(0..de_roi.len())].clone()
                        } else {
                            coups[rng.gen_range(0..coups.len())].clone()
                        };
                        let nous = sommet.turn();
                        let zone_avant = zone_du_camp(&sommet, nous);
                        pile.pousse(&eval, &sommet, &m);
                        let apres = sommet.play(&m).expect("coup légal");
                        if zone_du_camp(&apres, nous) != zone_avant {
                            changements += 1;
                        }
                        ucis.push(m.to_uci(CastlingMode::Standard).to_string());
                        positions.push(apres);
                    } else if positions.len() > 1 {
                        pile.depousse();
                        positions.pop();
                        ucis.pop();
                    } else {
                        break; // mat ou pat à la racine
                    }
                    let pos = positions.last().unwrap();
                    exige_parite_roi8(
                        net, &eval, &pile, pos, fen, &ucis, ucis.len(),
                        &format!("{contexte}, pas {pas}"),
                    );
                }
                println!("{contexte} : {pas_total} pas, {changements} changements de zone");
                if pas_total >= 2000 {
                    assert!(
                        changements > pas_total / 40,
                        "{contexte} : trop peu de changements de zone ({changements})"
                    );
                }
            }
        }
    }
}
