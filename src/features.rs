//! Encodage d'une position en vecteur de caractéristiques pour le réseau de valeur.
//!
//! CONVENTION CENTRALE (à respecter partout) : l'encodage est fait du point de vue
//! du TRAIT (side to move). Si les noirs jouent, on applique un miroir vertical du
//! plateau (case ^ 56) et on échange les couleurs, si bien que le réseau voit
//! toujours « ses » pièces comme les blanches. La sortie du réseau est donc
//! l'espérance de gain POUR LE TRAIT, dans [-1, 1].

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Position};

/// 768 = 12 types de pièces (6 nôtres puis 6 adverses, ordre P,N,B,R,Q,K) × 64 cases
/// + 4 droits de roque (notre O-O, notre O-O-O, leur O-O, leur O-O-O)
/// + 1 indicateur « une prise en passant est possible ».
pub const N_FEATURES: usize = 773;

/// Remplit `out` (longueur N_FEATURES, la fonction fait elle-même le remise à zéro)
/// avec l'encodage de `pos` du point de vue du trait.
/// Index d'une pièce : plan * 64 + case_vue_par_le_trait, où
/// plan ∈ [0,5] = nos P,N,B,R,Q,K et plan ∈ [6,11] = leurs P,N,B,R,Q,K,
/// et case_vue_par_le_trait = case ^ 56 si les noirs sont au trait.
pub fn encode(pos: &Chess, out: &mut [f32]) {
    assert_eq!(out.len(), N_FEATURES, "encode: tampon de sortie de mauvaise taille");
    out.fill(0.0);

    let nous = pos.turn();
    // Miroir vertical si les noirs sont au trait : le réseau voit toujours
    // « ses » pièces partir du bas du plateau.
    let miroir = nous == Color::Black;

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
        out[plan * 64 + idx_case] = 1.0;
    }

    // Droits de roque, ordre : notre O-O, notre O-O-O, leur O-O, leur O-O-O.
    let eux = nous.other();
    let roques = pos.castles();
    if roques.has(nous, CastlingSide::KingSide) {
        out[768] = 1.0;
    }
    if roques.has(nous, CastlingSide::QueenSide) {
        out[769] = 1.0;
    }
    if roques.has(eux, CastlingSide::KingSide) {
        out[770] = 1.0;
    }
    if roques.has(eux, CastlingSide::QueenSide) {
        out[771] = 1.0;
    }

    // Indicateur en passant : une prise en passant est réellement jouable
    // (mode Legal, pas seulement une case cible annoncée dans la FEN).
    if pos.ep_square(EnPassantMode::Legal).is_some() {
        out[772] = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;

    /// Analyse une FEN en position jouable (mode de roque standard).
    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    #[test]
    fn position_initiale_32_pieces_et_drapeaux() {
        let pos = Chess::default();
        let mut out = vec![0.0f32; N_FEATURES];
        encode(&pos, &mut out);

        // 32 pièces exactement dans les 768 premiers plans, valeurs 0/1.
        let somme: f32 = out[..768].iter().sum();
        assert_eq!(somme, 32.0);
        assert!(out.iter().all(|&v| v == 0.0 || v == 1.0));

        // 16 pièces à nous, 16 à eux.
        let a_nous: f32 = out[..6 * 64].iter().sum();
        let a_eux: f32 = out[6 * 64..768].iter().sum();
        assert_eq!(a_nous, 16.0);
        assert_eq!(a_eux, 16.0);

        // Quelques cases précises : notre pion e2 (plan 0, case 12),
        // notre roi e1 (plan 5, case 4), leur dame d8 (plan 10, case 59).
        assert_eq!(out[0 * 64 + 12], 1.0);
        assert_eq!(out[5 * 64 + 4], 1.0);
        assert_eq!(out[10 * 64 + 59], 1.0);

        // Les 4 droits de roque sont présents, pas d'en passant.
        assert_eq!(&out[768..772], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(out[772], 0.0);
    }

    #[test]
    fn symetrie_miroir_blancs_noirs() {
        // La position de départ est son propre miroir (vertical + échange des
        // couleurs) : son encodage vu des blancs doit donc être identique à
        // l'encodage de la même position avec les noirs au trait.
        let blancs = Chess::default();
        let noirs = pos_de_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");

        let mut vu_blancs = vec![0.0f32; N_FEATURES];
        let mut vu_noirs = vec![0.0f32; N_FEATURES];
        encode(&blancs, &mut vu_blancs);
        encode(&noirs, &mut vu_noirs);
        assert_eq!(vu_blancs, vu_noirs);
    }

    #[test]
    fn symetrie_miroir_position_asymetrique() {
        // Position asymétrique et sa version miroir (plateau retourné,
        // couleurs échangées, trait opposé) : encodages identiques attendus.
        let vue_a = pos_de_fen("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3");
        let vue_b = pos_de_fen("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3");

        let mut enc_a = vec![0.0f32; N_FEATURES];
        let mut enc_b = vec![0.0f32; N_FEATURES];
        encode(&vue_a, &mut enc_a);
        encode(&vue_b, &mut enc_b);
        assert_eq!(enc_a, enc_b);
    }

    #[test]
    fn drapeau_en_passant_legal_seulement() {
        // Case e3 annoncée mais aucun pion noir ne peut prendre : drapeau à 0.
        let sans_prise = pos_de_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
        let mut out = vec![0.0f32; N_FEATURES];
        encode(&sans_prise, &mut out);
        assert_eq!(out[772], 0.0);

        // Ici le pion e5 peut réellement prendre d6 en passant : drapeau à 1.
        let avec_prise =
            pos_de_fen("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
        encode(&avec_prise, &mut out);
        assert_eq!(out[772], 1.0);
    }

    #[test]
    fn droits_de_roque_perspective_du_trait() {
        // Les blancs n'ont plus que le grand roque, les noirs que le petit,
        // et c'est aux noirs de jouer : « notre O-O » = celui des noirs.
        let pos = pos_de_fen("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/R3KBNR b Qk - 0 1");
        let mut out = vec![0.0f32; N_FEATURES];
        encode(&pos, &mut out);
        // Ordre : notre O-O (noirs k = oui), notre O-O-O (noirs q = non),
        // leur O-O (blancs K = non), leur O-O-O (blancs Q = oui).
        assert_eq!(&out[768..772], &[1.0, 0.0, 0.0, 1.0]);
    }
}
