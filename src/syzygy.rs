//! Tables de finales Syzygy (3-4-5 pièces) : chargement du dossier
//! engines/syzygy et conversion des verdicts WDL/DTZ à l'échelle des scores
//! de la recherche (src/search.rs).
//!
//! Deux usages par la recherche (champ `Recherche::syzygy`, None par défaut =
//! comportement historique bit à bit) :
//! - à la RACINE, ≤ 5 pièces : le coup est joué par DTZ (`coup_racine`) —
//!   conversion parfaite, règle des 50 coups comprise, zéro nœud cherché ;
//! - DANS L'ARBRE, aux nœuds atteints par un coup zéroteur (prise ou coup de
//!   pion : halfmoves == 0) : verdict WDL exact (`sonde_noeud`), converti en
//!   score plat hors zone de mat.
//!
//! RÈGLE DES 50 COUPS (sémantique DTZ50) : les tables distinguent le gain
//! inconditionnel de la « cursed win » (gain frustrable par la règle) et la
//! perte du « blessed loss » (perte sauvable par la règle). Ces deux
//! dernières SONT des nulles en jeu réel : converties à 0.0 ici.

use shakmaty::{Chess, Move, Position};
use shakmaty_syzygy::{AmbiguousWdl, Tablebase, Wdl};

/// Tables Syzygy pour les échecs standard.
pub type Tables = Tablebase<Chess>;

/// Nombre de pièces couvert par les tables installées (jeu 3-4-5 complet).
pub const PIECES_MAX: usize = 5;

/// Score d'un GAIN de table, à l'échelle de search.rs : au-dessus de toute
/// évaluation réseau (|v| <= 1), en dessous de tout MAT trouvé par la
/// recherche (les mats vivent dans ±[872, 1000], le seuil SEUIL_MAT est à
/// 800) — un gain de table vaut mieux que la meilleure éval, mais un mat
/// DÉMONTRÉ dans l'arbre reste préféré. Constante PLATE (pas d'ajustement au
/// ply, contrairement aux mats) : le score est indépendant du chemin et se
/// stocke donc en TT sans la conversion réservée aux mats ; la PROGRESSION
/// vers le mat est garantie par le jeu DTZ à la racine, pas par les scores
/// internes.
pub const SCORE_TB: f32 = 500.0;

/// Charge toutes les tables (.rtbw/.rtbz) d'un dossier. Rend les tables et
/// le nombre de fichiers reconnus (3-4-5 complet : 290). Erreur si le
/// dossier est illisible ou vide de tables — message clair pour les harnais,
/// qui préfèrent échouer tôt que jouer sans l'assurance demandée.
pub fn charge(dossier: &str) -> Result<(Tables, usize), String> {
    let mut tables = Tablebase::new();
    let n = tables
        .add_directory(dossier)
        .map_err(|e| format!("dossier illisible : {e}"))?;
    if n == 0 {
        return Err("aucune table Syzygy (.rtbw/.rtbz) trouvée".to_string());
    }
    Ok((tables, n))
}

/// WDL (côté trait) → score de recherche. Voir l'en-tête : cursed win et
/// blessed loss valent NULLE (règle des 50 coups, sémantique DTZ50).
fn score_wdl(wdl: Wdl) -> f32 {
    match wdl {
        Wdl::Win => SCORE_TB,
        Wdl::Loss => -SCORE_TB,
        Wdl::Draw | Wdl::CursedWin | Wdl::BlessedLoss => 0.0,
    }
}

/// WDL « ambigu » (compteur des 50 coups non nul + arrondi DTZ, cas de la
/// racine) : mêmes conversions ; l'ambiguïté résiduelle (MaybeWin/MaybeLoss)
/// est rabattue sur la nulle par prudence — le coup DTZ joué reste optimal,
/// seul le score annoncé est conservateur.
fn score_wdl_ambigu(wdl: AmbiguousWdl) -> f32 {
    match wdl {
        AmbiguousWdl::Win => SCORE_TB,
        AmbiguousWdl::Loss => -SCORE_TB,
        AmbiguousWdl::Draw
        | AmbiguousWdl::CursedWin
        | AmbiguousWdl::BlessedLoss
        | AmbiguousWdl::MaybeWin
        | AmbiguousWdl::MaybeLoss => 0.0,
    }
}

/// Sonde WDL d'un NŒUD de recherche atteint par un coup zéroteur (le point
/// d'appel dans negamax garantit halfmoves == 0 — contrat de
/// `probe_wdl_after_zeroing`, qui rend alors un verdict EXACT vis-à-vis de
/// la règle des 50 coups avec les seules tables WDL). Toute erreur (droits
/// de roque résiduels, table manquante, E/S) vaut « pas d'information » :
/// la recherche continue normalement, jamais de panique.
pub fn sonde_noeud(tables: &Tables, pos: &Chess) -> Option<f32> {
    debug_assert_eq!(pos.halfmoves(), 0, "sonde réservée aux coups zéroteurs");
    tables.probe_wdl_after_zeroing(pos).ok().map(score_wdl)
}

/// Coup de RACINE par DTZ : le meilleur coup au sens des tables — préserve
/// le verdict et privilégie les coups zéroteurs (algorithme `best_move` du
/// crate). Nuance 50 coups : le compteur réel n'entre que dans le SCORE
/// annoncé (probe_wdl → WDL ambigu, rabattu prudemment sur la nulle) ; le
/// CHOIX du coup, lui, est DTZ-optimal sans lecture du compteur, avec
/// l'arrondi DTZ hors ligne principale assumé par la doc du crate
/// (imprécision confinée au voisinage de la frontière des 100 demi-coups).
/// Rendu avec le score WDL de la position à l'échelle de la recherche.
/// None : table manquante, droits de roque résiduels, ou aucun coup légal
/// (position terminale — la recherche rend son verdict exact).
pub fn coup_racine(tables: &Tables, pos: &Chess) -> Option<(Move, f32)> {
    let (coup, _dtz) = tables.best_move(pos).ok()??;
    let wdl = tables.probe_wdl(pos).ok()?;
    Some((coup, score_wdl_ambigu(wdl)))
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::N_FEATURES;
    use crate::nn::Mlp;
    use crate::search::{Limites, Recherche, SCORE_MAT};
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Color, Position, Square};
    use std::sync::Arc;

    /// Dossier des tables téléchargées (relatif à la racine du crate, le cwd
    /// de cargo test). Tests SAUTÉS proprement s'il est absent (machine sans
    /// tables) — sur le poste du chantier il est présent et ils tournent.
    const DOSSIER: &str = "engines/syzygy";

    fn tables() -> Option<Arc<Tables>> {
        if !std::path::Path::new(DOSSIER).is_dir() {
            println!("tables absentes ({DOSSIER}) : test sauté");
            return None;
        }
        let (t, n) = charge(DOSSIER).expect("chargement des tables");
        assert!(n >= 290, "jeu 3-4-5 complet attendu (290 fichiers), {n} reconnus");
        Some(Arc::new(t))
    }

    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    /// Réseau linéaire nul (éval constante 0.0) : les tests Syzygy ne mesurent
    /// que la recherche et les tables, et le profil dev reste rapide.
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

    /// La conversion des verdicts (aucune table requise) : gains/pertes aux
    /// bornes prévues, et TOUTES les nuances de nulle des 50 coups à 0.0.
    #[test]
    fn conversion_wdl_vers_score() {
        assert_eq!(score_wdl(Wdl::Win), SCORE_TB);
        assert_eq!(score_wdl(Wdl::Loss), -SCORE_TB);
        assert_eq!(score_wdl(Wdl::Draw), 0.0);
        assert_eq!(score_wdl(Wdl::CursedWin), 0.0);
        assert_eq!(score_wdl(Wdl::BlessedLoss), 0.0);
        // L'échelle promise par le commentaire de SCORE_TB : au-dessus des
        // évals réseau, sous le plancher des mats de la recherche.
        assert!(SCORE_TB > 1.0 && SCORE_TB < SCORE_MAT - 200.0);
        // Côté racine : l'ambiguïté d'arrondi est rabattue sur la nulle.
        assert_eq!(score_wdl_ambigu(AmbiguousWdl::MaybeWin), 0.0);
        assert_eq!(score_wdl_ambigu(AmbiguousWdl::MaybeLoss), 0.0);
        assert_eq!(score_wdl_ambigu(AmbiguousWdl::Win), SCORE_TB);
        assert_eq!(score_wdl_ambigu(AmbiguousWdl::Loss), -SCORE_TB);
    }

    /// KQvK : depuis la racine, les DEUX camps joués par DTZ (le défenseur
    /// aussi : best_move rend la meilleure défense) → mat FORCÉ délivré, sans
    /// un seul nœud de recherche. Le premier verdict est un gain de table net.
    #[test]
    fn kqvk_mat_force_par_dtz_a_la_racine() {
        let Some(tb) = tables() else { return };
        let mut r = Recherche::new(reseau_nul(), 12);
        r.syzygy = Some(tb);
        let mut pos = pos_de_fen("4k3/8/8/8/8/8/8/4KQ2 w - - 0 1");

        let premier = r.cherche(&pos, limites_noeuds(400));
        assert_eq!(premier.score, SCORE_TB, "gain de table attendu d'emblée");
        assert_eq!(premier.noeuds, 0, "coup par DTZ : zéro nœud cherché");
        assert_eq!(premier.profondeur, 0);

        let mut plis = 0u32;
        while !pos.legal_moves().is_empty() && plis < 60 {
            let res = r.cherche(&pos, limites_noeuds(400));
            let coup = res.coup.expect("coup légal");
            pos = pos.play(&coup).expect("coup légal");
            plis += 1;
        }
        assert!(
            pos.legal_moves().is_empty() && pos.is_check() && pos.turn() == Color::Black,
            "mat des noirs attendu en < 60 plis (plis joués : {plis}, fen : {})",
            Fen::from_position(pos.clone(), shakmaty::EnPassantMode::Legal)
        );
    }

    /// KRvKR (tours défendues, aucune prise) : nulle inconditionnelle au WDL,
    /// donc 0.0 à l'échelle de la recherche — à la sonde de nœud comme à la
    /// racine.
    #[test]
    fn krvkr_nulle_wdl() {
        let Some(tb) = tables() else { return };
        let pos = pos_de_fen("1k6/1r6/8/8/8/8/6R1/6K1 w - - 0 1");
        assert_eq!(tb.probe_wdl_after_zeroing(&pos).expect("sonde WDL"), Wdl::Draw);
        assert_eq!(sonde_noeud(&tb, &pos), Some(0.0));
        let (_, score) = coup_racine(&tb, &pos).expect("coup de table");
        assert_eq!(score, 0.0);
    }

    /// « Cursed win » (gain frustrable par la règle des 50 coups) → NULLE,
    /// c'est LA subtilité de la sémantique DTZ50. Position vérifiée contre
    /// les tables locales (diag_scan_cursed_win) ET le miroir lichess
    /// (DTZ 102) : KNNvKP pion e5, la victoire des cavaliers exige plus de
    /// 100 demi-coups sans zérotage (|DTZ| > 100).
    #[test]
    fn cursed_win_vaut_nulle() {
        let Some(tb) = tables() else { return };
        let pos = pos_de_fen("8/8/8/4p3/8/8/8/K1k2N1N w - - 0 1");
        let wdl = tb.probe_wdl_after_zeroing(&pos).expect("sonde WDL");
        assert_eq!(wdl, Wdl::CursedWin, "le scénario exige une cursed win");
        assert_eq!(
            sonde_noeud(&tb, &pos),
            Some(0.0),
            "une cursed win doit valoir NULLE (règle des 50 coups)"
        );
    }

    /// Transition vue par la RECHERCHE : 6 pièces à la racine (hors tables),
    /// et UNE SEULE prise — Dxd8, la dame noire non défendue — fait entrer la
    /// ligne en territoire 5 pièces. Le nœud fils (coup zéroteur) est sondé
    /// au WDL (KQ contre K+2 pions : perte de table certaine pour le trait
    /// noir), le verdict exact remonte : la racine choisit la prise avec le
    /// score de gain de table — que le réseau (éval nulle partout ici)
    /// n'aurait jamais fourni.
    #[test]
    fn transition_prise_sondee_dans_l_arbre() {
        let Some(tb) = tables() else { return };
        let mut r = Recherche::new(reseau_nul(), 14);
        r.syzygy = Some(tb);
        // Blancs : Dd1, Re1 ; noirs : Dd8 (colonne d ouverte, non défendue),
        // Rg7, pions g6 et h7 — 6 pièces.
        let pos = pos_de_fen("3q4/6kp/6p1/8/8/8/8/3QK3 w - - 0 1");
        let res = r.cherche(
            &pos,
            Limites { max_noeuds: 0, max_profondeur: 2, movetime_ms: 0 },
        );
        assert_eq!(res.score, SCORE_TB, "le gain de table doit remonter à la racine");
        assert_eq!(
            res.coup.expect("coup légal").to(),
            Square::D8,
            "la prise Dxd8 (entrée en territoire de tables gagnant) attendue"
        );
    }

    /// COUPERET FINALES (harnais opérateur, ignoré par défaut) : 20 positions
    /// de finales CONNUES jouées par le moteur AVEC les tables (limite de
    /// recherche courte : 1000 nœuds), 20/20 exigé. Trois exigences :
    /// - le VERDICT racine est exact (gain → SCORE_TB, nulle → 0.0, et les
    ///   cursed wins valent NULLE — règle des 50 coups) ;
    /// - un GAIN se CONVERTIT : les deux camps joués par le moteur → mat
    ///   effectivement délivré par le camp fort (≤ 220 plis) ;
    /// - une NULLE se TIENT : 60 plis joués des deux côtés sans que le score
    ///   ne quitte 0.0 (fin prématurée tolérée si pat/matériel insuffisant).
    /// Lancer :
    /// cargo test --lib syzygy::tests::couperet_finales_20 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn couperet_finales_20() {
        let tb = tables().expect("couperet : tables 3-4-5 requises");
        // (nom, fen, gain attendu pour le trait ? — false = nulle)
        let suite: &[(&str, &str, bool)] = &[
            // Gains KQvK (mat à distance variable).
            ("KQvK proche", "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1", true),
            ("KQvK distance", "k7/8/8/8/8/8/1Q6/7K w - - 0 1", true),
            ("KQvK centre", "8/8/8/3k4/8/8/8/KQ6 w - - 0 1", true),
            ("KQvK coin", "8/8/1k6/8/8/8/6Q1/K7 w - - 0 1", true),
            // Gains KRvK.
            ("KRvK coin", "k7/8/8/8/8/8/8/1R4K1 w - - 0 1", true),
            ("KRvK centre", "8/8/8/4k3/8/8/8/R3K3 w - - 0 1", true),
            ("KRvK distance", "7k/8/8/8/8/8/8/R6K w - - 0 1", true),
            // Gains KPvK (roi devant sur la 6e ; course de promotion ;
            // règle du carré).
            ("KPvK roi 6e", "4k3/8/4K3/4P3/8/8/8/8 w - - 0 1", true),
            ("KPvK course", "8/8/8/8/8/8/P7/K6k w - - 0 1", true),
            ("KPvK roi 6e loin", "6k1/8/6K1/8/8/8/6P1/8 w - - 0 1", true),
            ("KPvK carre", "8/8/8/8/4P3/4K3/8/6k1 w - - 0 1", true),
            // Cursed wins (gain frustré par la règle des 50 coups → NULLE) —
            // positions vérifiées par diag_scan_cursed_win (DTZ 102 et 116).
            ("KNNvKP cursed e5", "8/8/8/4p3/8/8/8/K1k2N1N w - - 0 1", false),
            ("KNNvKP cursed h5", "8/8/8/7p/8/8/8/K1k2N1N w - - 0 1", false),
            // Nulles KRvKR.
            ("KRvKR", "1k6/1r6/8/8/8/8/6R1/6K1 w - - 0 1", false),
            ("KRvKR bis", "8/8/4k3/8/r7/8/4K3/7R w - - 0 1", false),
            // Nulles de matériel insuffisant.
            ("KBvK", "8/8/8/8/8/8/8/KB5k w - - 0 1", false),
            ("KBvK bis", "7k/8/8/8/8/3B4/8/K7 w - - 0 1", false),
            ("KNvK", "8/8/8/4k3/8/8/8/KN6 w - - 0 1", false),
            // Nulles KPvK (pion tour, roi défenseur au coin ; roi défenseur
            // devant le pion, roi fort derrière).
            ("KPvK pion tour", "7k/8/8/8/8/8/7P/7K w - - 0 1", false),
            ("KPvK bloque", "8/8/8/8/8/4k3/4P3/4K3 w - - 0 1", false),
        ];
        assert_eq!(suite.len(), 20, "le couperet exige 20 positions");
        let mut ok = 0;
        for (nom, fen, gain) in suite {
            let mut r = Recherche::new(reseau_nul(), 12);
            r.syzygy = Some(tb.clone());
            let mut pos = pos_de_fen(fen);
            let attendu = if *gain { SCORE_TB } else { 0.0 };
            let res = r.cherche(&pos, limites_noeuds(1000));
            let verdict_ok = res.score == attendu;
            // Conversion (gains) / tenue (nulles) : les DEUX camps joués par
            // le moteur armé des tables.
            let mut jeu_ok = true;
            if *gain {
                let mut plis = 0u32;
                while !pos.legal_moves().is_empty() && plis < 220 {
                    let c = r.cherche(&pos, limites_noeuds(1000)).coup.expect("coup légal");
                    pos = pos.play(&c).expect("coup légal");
                    plis += 1;
                }
                // Mat délivré par les blancs (camp fort de toute la suite).
                jeu_ok = pos.legal_moves().is_empty()
                    && pos.is_check()
                    && pos.turn() == Color::Black;
            } else {
                for _ in 0..60 {
                    if pos.legal_moves().is_empty() {
                        // Pat toléré (c'est une nulle) — jamais un mat.
                        jeu_ok = !pos.is_check();
                        break;
                    }
                    let res = r.cherche(&pos, limites_noeuds(1000));
                    if res.score != 0.0 {
                        jeu_ok = false;
                        break;
                    }
                    pos = pos.play(&res.coup.expect("coup légal")).expect("coup légal");
                }
            }
            let bon = verdict_ok && jeu_ok;
            ok += u32::from(bon);
            println!(
                "{} {nom} : verdict {} (attendu {attendu}), jeu {}",
                if bon { "OK " } else { "ECHEC" },
                res.score,
                if jeu_ok { "tenu" } else { "fautif" }
            );
        }
        println!("couperet finales : {ok}/20");
        assert_eq!(ok, 20, "couperet finales : {ok}/20");
    }

    /// Diagnostic (ignoré par défaut) : balaye une famille KNNvKP — rois et
    /// cavaliers fixes sur la première rangée, pion noir baladé — et imprime
    /// les « cursed wins » trouvées avec leur DTZ. Sert à choisir/vérifier la
    /// position du test cursed_win_vaut_nulle.
    /// Lancer : cargo test --lib syzygy::tests::diag -- --ignored --nocapture
    #[test]
    #[ignore]
    fn diag_scan_cursed_win() {
        let Some(tb) = tables() else { return };
        let mut trouvees = 0;
        for ci in 0..8usize {
            for rangee in 2..=6usize {
                // Blancs Ka1 Cf1 Ch1, noir Rc1, pion noir en (colonne ci,
                // rangée) : FEN reconstruite rang par rang (rang 8 en tête).
                let mut rangs = vec!["8".to_string(); 8];
                rangs[7] = "K1k2N1N".to_string();
                let mut rang = String::new();
                if ci > 0 {
                    rang.push_str(&ci.to_string());
                }
                rang.push('p');
                if ci < 7 {
                    rang.push_str(&(7 - ci).to_string());
                }
                rangs[8 - rangee] = rang;
                let fen = format!("{} w - - 0 1", rangs.join("/"));
                let Ok(setup) = fen.parse::<Fen>() else { continue };
                let Ok(pos) = setup.into_position::<Chess>(CastlingMode::Standard) else {
                    continue;
                };
                if let Ok(wdl) = tb.probe_wdl_after_zeroing(&pos) {
                    if wdl == Wdl::CursedWin {
                        trouvees += 1;
                        println!("cursed win : {fen} (dtz {:?})", tb.probe_dtz(&pos));
                    }
                }
            }
        }
        println!("{trouvees} cursed win(s) dans la famille balayée");
    }
}
