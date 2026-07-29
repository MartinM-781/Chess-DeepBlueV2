//! Caractéristiques ROI-RELATIVES par zones (à la NNUE), représentation CREUSE.
//!
//! POURQUOI : l'encodage classique (`features::encode`, 773 entrées) ignore OÙ
//! se trouve le roi — la même table pièce-case sert que le roi soit au centre
//! ou roqué, ce qui plafonne le réseau (~1550 Elo). Ici, chaque plan
//! pièce-case est DUPLIQUÉ par zone du roi du camp de la perspective : le
//! réseau apprend des tables différentes selon la position de SON roi (le
//! principe des « king buckets » NNUE), pour un coût d'inférence inchangé
//! grâce à la représentation CREUSE : une position n'active que
//! nombre_de_pièces (≤ 32) + drapeaux (≤ 5), soit au plus 37 indices sur 6149.
//!
//! CONVENTION DE PERSPECTIVE : identique à `features::encode` — miroir
//! vertical (case ^ 56) et échange des couleurs si la perspective est noire,
//! si bien que la perspective voit toujours « ses » pièces partir du bas du
//! plateau. `actifs` encode pour le TRAIT ; `actifs_perspective` pour un camp
//! imposé (les accumulateurs NNUE entretiennent les DEUX perspectives, chacune
//! conditionnée par la zone de SON PROPRE roi).

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Position};

/// Nombre de zones possibles pour le roi DU CAMP DE LA PERSPECTIVE :
/// 4 quadrants × 2 moitiés de quadrant (voir `zone_roi` pour la définition
/// exacte de la partition).
pub const N_ZONES_ROI: usize = 8;

/// 6149 = 8 zones × 768 plans pièce-case (12 pièces × 64 cases, mêmes plans
/// que `features::encode`) + 5 scalaires en QUEUE (indices 6144..6149) :
/// notre O-O, notre O-O-O, leur O-O, leur O-O-O, prise en passant légale.
pub const N_FEATURES_ROI: usize = N_ZONES_ROI * 768 + 5;

/// Zone (0..8) de la case du roi, EXPRIMÉE DANS LA PERSPECTIVE de son camp
/// (donc après l'éventuel miroir `case ^ 56` : la rangée 1 est toujours la
/// rangée de départ du camp considéré).
///
/// Partition exacte des 64 cases, « 4 quadrants × 2 moitiés » :
/// - quadrant = aile (files a–d / e–h) × camp (rangées 1–4 / 5–8) ;
/// - moitié   = paire de rangées basse/haute À L'INTÉRIEUR du quadrant
///   (rangées 1–2 vs 3–4 dans notre camp, 5–6 vs 7–8 dans le camp adverse).
///
/// Soit, de façon équivalente : files a–d / e–h × bandes de rangées
/// {1–2, 3–4, 5–6, 7–8}. Chaque zone contient exactement 8 cases (4 files ×
/// 2 rangées) :
///
/// ```text
///   rangées (perspective)   files a–d   files e–h
///   7–8 (fond adverse)          5           7
///   5–6                         4           6
///   3–4                         1           3
///   1–2 (notre camp)            0           2
/// ```
///
/// Un roi « normal » (resté sur ses deux premières rangées) tombe donc en
/// zone 0 (aile dame) ou 2 (aile roi) — les deux zones les plus visitées —
/// et tout roi aventuré plus haut change de bloc de 768 features.
pub fn zone_roi(case_roi_perspective: usize) -> usize {
    assert!(
        case_roi_perspective < 64,
        "zone_roi: case hors du plateau : {case_roi_perspective}"
    );
    let colonne = case_roi_perspective & 7; // 0..8 = files a..h
    let rangee = case_roi_perspective >> 3; // 0..8 = rangées 1..8 (perspective)
    let quadrant = (rangee / 4) * 2 + colonne / 4; // 0..4
    let moitie = (rangee / 2) % 2; // paire de rangées basse (0) / haute (1) du quadrant
    quadrant * 2 + moitie
}

/// Remplit `sortie` (vidée d'abord) avec les indices actifs de `pos` pour la
/// perspective demandée (`blanc` = vrai → point de vue des blancs). Utilisée
/// par les accumulateurs NNUE qui entretiennent les DEUX perspectives : chaque
/// perspective est conditionnée par la zone de SON PROPRE roi.
///
/// Encodage (mêmes conventions de miroir que `features::encode`) :
/// - si la perspective est noire, chaque case subit le miroir vertical
///   `case ^ 56` et les couleurs sont échangées (« nos » pièces = les noires) ;
/// - indice d'une pièce : `zone * 768 + plan * 64 + case_vue`, avec
///   plan ∈ [0,5] = nos P,N,B,R,Q,K et plan ∈ [6,11] = leurs P,N,B,R,Q,K,
///   et zone = `zone_roi(case_vue_de_NOTRE_roi)` — la MÊME zone pour toutes
///   les pièces de la position ;
/// - scalaires (actifs seulement si vrais) : 6144 notre O-O, 6145 notre O-O-O,
///   6146 leur O-O, 6147 leur O-O-O, 6148 prise en passant LÉGALE. « Notre »
///   désigne le camp de la PERSPECTIVE. Le drapeau en passant décrit l'état
///   réel de la position (seul le camp au trait peut prendre en passant) : il
///   est identique pour les deux perspectives, comme dans `features::encode`.
///
/// Les indices sont produits dans un ordre déterministe (pièces dans l'ordre
/// du plateau, puis scalaires croissants) mais PAS globalement triés ; ils
/// sont uniques (au plus 37 = 32 pièces + 5 drapeaux).
pub fn actifs_perspective(pos: &Chess, blanc: bool, sortie: &mut Vec<u16>) {
    sortie.clear();

    let nous = if blanc { Color::White } else { Color::Black };
    let miroir = !blanc;

    // Zone de NOTRE roi (celui de la perspective), vue dans la perspective.
    let roi = pos
        .board()
        .king_of(nous)
        .expect("position légale : chaque camp a un roi");
    let case_roi = if miroir {
        usize::from(roi) ^ 56
    } else {
        usize::from(roi)
    };
    let zone = zone_roi(case_roi);
    let base_zone = zone * 768;

    // Plans de pièces : Role vaut 1..=6 dans l'ordre P,N,B,R,Q,K → plan role-1
    // pour nos pièces, 6 + (role-1) pour celles de l'adversaire.
    for (case, piece) in pos.board().iter() {
        let idx_case = if miroir {
            usize::from(case) ^ 56
        } else {
            usize::from(case)
        };
        let plan = if piece.color == nous {
            usize::from(piece.role) - 1
        } else {
            6 + usize::from(piece.role) - 1
        };
        let indice = base_zone + plan * 64 + idx_case;
        debug_assert!(indice < N_ZONES_ROI * 768);
        sortie.push(indice as u16);
    }

    // Scalaires en queue, mêmes définitions et même ordre que
    // `features::encode` (768..773 là-bas, 6144..6149 ici).
    let eux = nous.other();
    let roques = pos.castles();
    let base_scalaires = (N_ZONES_ROI * 768) as u16; // 6144
    if roques.has(nous, CastlingSide::KingSide) {
        sortie.push(base_scalaires);
    }
    if roques.has(nous, CastlingSide::QueenSide) {
        sortie.push(base_scalaires + 1);
    }
    if roques.has(eux, CastlingSide::KingSide) {
        sortie.push(base_scalaires + 2);
    }
    if roques.has(eux, CastlingSide::QueenSide) {
        sortie.push(base_scalaires + 3);
    }
    if pos.ep_square(EnPassantMode::Legal).is_some() {
        sortie.push(base_scalaires + 4);
    }
}

/// Indices actifs de `pos` du point de vue du TRAIT (la perspective de
/// `features::encode`) : équivaut à `actifs_perspective(pos, trait_blanc, …)`.
/// C'est l'entrée du chemin creux (`Mlp::forward_actifs`,
/// `Mlp::train_batch_actifs`) pour un réseau au schéma `RoiZones8`.
///
/// # Exemple complet : position après 1.e4 (trait aux noirs)
///
/// FEN `rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1`.
///
/// **Perspective du trait (noirs)** — ce que renvoie `actifs` : miroir
/// vertical et couleurs échangées.
/// - Roi noir en e8 (case 60) → vu `60 ^ 56 = 4` (« e1 » de sa perspective)
///   → `zone_roi(4) = 2` : toutes les pièces tombent dans le bloc
///   `[2·768, 3·768) = [1536, 2304)`.
/// - Notre pion e7 (case 52, vu 12) : `1536 + 0·64 + 12 = 1548`.
/// - Leur pion e4 (case 28, vu 36, « e5 » de notre point de vue) :
///   `1536 + 6·64 + 36 = 1956`.
/// - Leur roi e1 (case 4, vu 60) : `1536 + 11·64 + 60 = 2300`.
/// - Scalaires : les 4 roques → 6144, 6145, 6146, 6147. La case e3 est
///   annoncée dans la FEN mais AUCUNE prise en passant n'est légale → 6148
///   inactif (convention `EnPassantMode::Legal`, comme `features::encode`).
/// - Total : 32 pièces + 4 roques = 36 indices actifs.
///
/// **Perspective blanche** — `actifs_perspective(pos, true, …)` : pas de
/// miroir.
/// - Roi blanc en e1 (case 4) → `zone_roi(4) = 2` aussi (les deux rois sont
///   encore chez eux) : même bloc `[1536, 2304)`.
/// - Notre pion e4 (case 28) : `1536 + 0·64 + 28 = 1564`.
/// - Leur roi e8 (case 60) : `1536 + 11·64 + 60 = 2300` — le même indice que
///   le roi blanc vu des noirs : la position de départ est presque symétrique.
/// - Mêmes 4 roques, même absence d'en passant : 36 indices également.
pub fn actifs(pos: &Chess, sortie: &mut Vec<u16>) {
    actifs_perspective(pos, pos.turn() == Color::White, sortie);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;

    /// Analyse une FEN en position jouable (mode de roque standard).
    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    /// `n` positions atteintes par coups légaux aléatoires (graine fixe),
    /// en repartant du début quand la partie s'enlise ou se termine.
    fn positions_aleatoires(n: usize, graine: u64) -> Vec<Chess> {
        let mut rng = StdRng::seed_from_u64(graine);
        let mut sortie = Vec::with_capacity(n);
        let mut pos = Chess::default();
        let mut plis = 0usize;
        while sortie.len() < n {
            let coups = pos.legal_moves();
            if coups.is_empty()
                || plis >= 120
                || pos.is_insufficient_material()
                || pos.halfmoves() >= 100
            {
                pos = Chess::default();
                plis = 0;
                continue;
            }
            let coup = coups[rng.gen_range(0..coups.len())].clone();
            pos = pos.play(&coup).expect("coup légal");
            plis += 1;
            sortie.push(pos.clone());
        }
        sortie
    }

    #[test]
    fn zone_roi_partitionne_exactement_les_64_cases() {
        let mut effectifs = [0usize; N_ZONES_ROI];
        for case in 0..64 {
            let z = zone_roi(case);
            assert!(z < N_ZONES_ROI, "case {case} : zone {z} hors bornes");
            effectifs[z] += 1;
        }
        // Partition exacte : chaque case dans UNE zone, la somme des effectifs
        // couvre les 64 cases, et chaque zone en reçoit 8 (4 files × 2 rangées).
        assert_eq!(effectifs.iter().sum::<usize>(), 64);
        assert_eq!(effectifs, [8; N_ZONES_ROI]);

        // Points de repère de la table de la doc.
        assert_eq!(zone_roi(0), 0); // a1
        assert_eq!(zone_roi(2), 0); // c1
        assert_eq!(zone_roi(4), 2); // e1
        assert_eq!(zone_roi(6), 2); // g1
        assert_eq!(zone_roi(24), 1); // a4
        assert_eq!(zone_roi(36), 6); // e5
        assert_eq!(zone_roi(56), 5); // a8
        assert_eq!(zone_roi(63), 7); // h8
    }

    #[test]
    fn position_initiale_32_pieces_plus_4_roques() {
        let pos = Chess::default();
        let mut sortie = Vec::new();
        actifs(&pos, &mut sortie);

        // 32 pièces + 4 droits de roque, pas d'en passant : 36 indices actifs.
        assert_eq!(sortie.len(), 36);
        let mut tri = sortie.clone();
        tri.sort_unstable();
        tri.dedup();
        assert_eq!(tri.len(), 36, "indice actif en double");
        assert!(tri.iter().all(|&i| usize::from(i) < N_FEATURES_ROI));

        // Roi blanc en e1 (case 4) → zone 2 : TOUTES les pièces tombent dans
        // le bloc [2·768, 3·768) ; 16 à nous puis 16 à eux.
        let pieces: Vec<usize> = tri
            .iter()
            .map(|&i| usize::from(i))
            .filter(|&i| i < N_ZONES_ROI * 768)
            .collect();
        assert_eq!(pieces.len(), 32);
        assert!(pieces.iter().all(|&i| (1536..2304).contains(&i)));
        assert_eq!(pieces.iter().filter(|&&i| i < 1536 + 6 * 64).count(), 16);

        // Cases précises (mêmes repères que le test de features::encode,
        // décalés du bloc de zone) : notre pion e2, notre roi e1, leur dame d8.
        assert!(tri.contains(&1548)); // 1536 + 0·64 + 12
        assert!(tri.contains(&1860)); // 1536 + 5·64 + 4
        assert!(tri.contains(&2235)); // 1536 + 10·64 + 59

        // Les 4 roques présents, pas d'en passant.
        for k in 0..4u16 {
            assert!(tri.contains(&(6144 + k)));
        }
        assert!(!tri.contains(&6148));
    }

    #[test]
    fn perspectives_symetriques_sur_la_position_initiale() {
        // La position de départ est son propre miroir (vertical + échange des
        // couleurs) : les deux perspectives produisent le MÊME ensemble
        // d'indices, et `actifs` (trait = blancs) coïncide avec la perspective
        // blanche.
        let pos = Chess::default();
        let (mut blancs, mut noirs, mut du_trait) = (Vec::new(), Vec::new(), Vec::new());
        actifs_perspective(&pos, true, &mut blancs);
        actifs_perspective(&pos, false, &mut noirs);
        actifs(&pos, &mut du_trait);
        assert_eq!(du_trait, blancs);
        blancs.sort_unstable();
        noirs.sort_unstable();
        assert_eq!(blancs, noirs);
    }

    #[test]
    fn exemple_de_la_doc_apres_1_e4() {
        // Vérifie chiffre à chiffre l'exemple du doc-comment de `actifs`.
        let pos = pos_de_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");

        // Perspective du trait (noirs) : miroir, roi e8 vu en « e1 » → zone 2.
        let mut noirs = Vec::new();
        actifs(&pos, &mut noirs);
        assert_eq!(noirs.len(), 36);
        assert!(noirs.contains(&1548)); // notre pion e7, vu case 12
        assert!(noirs.contains(&1956)); // leur pion e4, vu case 36 (« e5 »)
        assert!(noirs.contains(&2300)); // leur roi e1, vu case 60
        assert!(!noirs.contains(&6148)); // e3 annoncée mais prise illégale

        // Perspective blanche : pas de miroir, roi e1 → zone 2 aussi.
        let mut blancs = Vec::new();
        actifs_perspective(&pos, true, &mut blancs);
        assert_eq!(blancs.len(), 36);
        assert!(blancs.contains(&1564)); // notre pion e4 (case 28)
        assert!(blancs.contains(&2300)); // leur roi e8 (case 60)

        // Les 4 droits de roque pour les deux perspectives.
        for liste in [&noirs, &blancs] {
            for k in 0..4u16 {
                assert!(liste.contains(&(6144 + k)));
            }
        }
    }

    #[test]
    fn coherence_avec_l_encodage_classique_773() {
        // Sur des positions aléatoires : en retirant le bloc de zone, les
        // indices actifs doivent EXACTEMENT reproduire les indices non nuls de
        // `features::encode` (même perspective du trait, mêmes plans, mêmes
        // drapeaux) — les deux encodages ne diffèrent que par la duplication
        // par zone du roi.
        let positions = positions_aleatoires(60, 20260726);
        let mut dense = vec![0.0f32; crate::features::N_FEATURES];
        let mut creux = Vec::new();
        for pos in &positions {
            crate::features::encode(pos, &mut dense);
            actifs(pos, &mut creux);

            // Toutes les pièces partagent la MÊME zone : celle du roi du trait.
            let zones: Vec<usize> = creux
                .iter()
                .map(|&i| usize::from(i))
                .filter(|&i| i < N_ZONES_ROI * 768)
                .map(|i| i / 768)
                .collect();
            assert!(zones.windows(2).all(|w| w[0] == w[1]));

            let mut depuis_creux: Vec<usize> = creux
                .iter()
                .map(|&i| {
                    let i = usize::from(i);
                    if i < N_ZONES_ROI * 768 {
                        i % 768 // retire le bloc de zone
                    } else {
                        768 + (i - N_ZONES_ROI * 768) // scalaires : 6144+k ↔ 768+k
                    }
                })
                .collect();
            depuis_creux.sort_unstable();
            let depuis_dense: Vec<usize> = dense
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == 1.0)
                .map(|(k, _)| k)
                .collect();
            assert_eq!(depuis_creux, depuis_dense);
        }
    }

    #[test]
    fn zone_change_quand_le_roi_traverse_une_frontiere() {
        // Roi blanc en e1 (zone 2, aile roi) puis en d1 (zone 0, aile dame) :
        // tout le bloc de pièces bascule de [1536, 2304) vers [0, 768).
        let roi_e1 = pos_de_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
        let roi_d1 = pos_de_fen("4k3/8/8/8/8/8/8/3K4 w - - 0 1");
        let (mut a, mut b) = (Vec::new(), Vec::new());
        actifs(&roi_e1, &mut a);
        actifs(&roi_d1, &mut b);
        assert_eq!(a.len(), 2); // deux rois, aucun droit de roque, pas d'ep
        assert_eq!(b.len(), 2);
        assert!(a.iter().all(|&i| (1536..2304).contains(&usize::from(i))));
        assert!(b.iter().all(|&i| usize::from(i) < 768));
    }
}
