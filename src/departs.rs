//! Positions de départ variées pour le self-play (et livre au plateau).
//!
//! Quatre familles de départs, tirées par `tirage` / `tirage_complet` :
//! - « ouverture:... » : une ligne de théorie RÉELLE du livre privé `LIGNES`,
//!   jouée intégralement depuis la position initiale (2 plis chauds ensuite) ;
//! - « finale:KRPvKR » etc. : une finale GÉNÉRÉE depuis un gabarit de matériel,
//!   pièces placées au hasard puis validées par shakmaty (0 pli chaud) ;
//! - « transition:... » : un MILIEU TARDIF généré (10 à 16 pièces, matériel
//!   équilibré à un pion près, rois abrités, aucune prise gagnante évidente
//!   au premier coup — vérification SEE), le territoire fin-de-milieu →
//!   finale où se jouent les conversions (0 pli chaud) ;
//! - « initiale » : la position de départ historique (8 plis chauds).
//!
//! Le PLATEAU (serve.exe) réutilise le même livre via `coup_du_livre` : si la
//! position courante est un préfixe exact d'une ou plusieurs lignes, l'IA joue
//! une continuation du livre tirée au hasard.

use std::collections::HashMap;
use std::sync::OnceLock;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use shakmaty::san::SanPlus;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{
    Board, CastlingMode, Chess, Color, EnPassantMode, File, Move, Position, Rank, Role, Setup,
    Square,
};

/// Position de départ d'une partie de self-play.
#[derive(Clone, Debug)]
pub struct Depart {
    /// La position, déjà validée par shakmaty.
    pub pos: Chess,
    /// « ouverture:italienne-giuoco-piano », « finale:KRPvKR », « initiale ».
    pub etiquette: &'static str,
    /// Plis joués à la température d'ouverture depuis cette position :
    /// 8 pour « initiale » (comportement historique), 2 pour une ouverture du
    /// livre (la théorie a déjà diversifié), 0 pour une finale.
    pub plis_chauds: u32,
}

/// Hachage zobrist de la position (même convention que selfplay/arena/serve :
/// en passant légal uniquement — les compteurs de coups n'y entrent pas, les
/// transpositions du plateau retombent donc bien sur les préfixes du livre).
fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

// ---------------------------------------------------------------------------
// Livre d'ouvertures
// ---------------------------------------------------------------------------

/// Les lignes du livre telles quelles — (étiquette, coups SAN). Exposées pour
/// les MATCHS : deux moteurs déterministes rejouent sinon la même partie à
/// chaque ronde, et l'usage des matchs de moteurs est d'imposer une ouverture
/// différente par ronde, la même jouée des DEUX couleurs.
pub fn lignes_du_livre() -> &'static [(&'static str, &'static str)] {
    LIGNES
}

/// Le livre : (étiquette, coups SAN séparés par des espaces). Théorie réelle,
/// 6 à 12 demi-coups par ligne. Toute ligne illégale fait paniquer le premier
/// accès au livre avec un message qui NOMME la ligne fautive (test dédié).
const LIGNES: &[(&str, &str)] = &[
    // --- Italienne ---
    ("ouverture:italienne-giuoco-piano", "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+"),
    ("ouverture:italienne-pianissimo", "e4 e5 Nf3 Nc6 Bc4 Bc5 d3 Nf6 c3 d6 O-O a6"),
    ("ouverture:italienne-deux-cavaliers", "e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Be7 O-O O-O"),
    ("ouverture:italienne-deux-cavaliers-ng5", "e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6"),
    ("ouverture:italienne-gambit-evans", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 d4 exd4"),
    ("ouverture:italienne-evans-refuse", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bb6 a4 a6"),
    ("ouverture:italienne-hongroise", "e4 e5 Nf3 Nc6 Bc4 Be7 d4 d6"),
    // --- Espagnole ---
    ("ouverture:espagnole-fermee", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5"),
    ("ouverture:espagnole-ouverte", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5"),
    ("ouverture:espagnole-echange", "e4 e5 Nf3 Nc6 Bb5 a6 Bxc6 dxc6 O-O f6 d4 exd4"),
    ("ouverture:espagnole-berlin", "e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6"),
    ("ouverture:espagnole-anti-berlin", "e4 e5 Nf3 Nc6 Bb5 Nf6 d3 Bc5 c3 O-O O-O d6"),
    ("ouverture:espagnole-steinitz", "e4 e5 Nf3 Nc6 Bb5 d6 d4 exd4 Nxd4 Bd7"),
    ("ouverture:espagnole-schliemann", "e4 e5 Nf3 Nc6 Bb5 f5 Nc3 fxe4 Nxe4 d5"),
    ("ouverture:espagnole-arkhangelsk", "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O b5 Bb3 Bb7"),
    // --- Sicilienne ---
    ("ouverture:sicilienne-najdorf", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6"),
    ("ouverture:sicilienne-najdorf-be2", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5"),
    ("ouverture:sicilienne-najdorf-anglaise", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5"),
    ("ouverture:sicilienne-najdorf-bg5", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Bg5 e6"),
    ("ouverture:sicilienne-dragon", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7"),
    ("ouverture:sicilienne-dragon-yougoslave", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 f3 Bg7"),
    ("ouverture:sicilienne-dragon-accelere", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 Nc3 Bg7 Be3 Nf6"),
    ("ouverture:sicilienne-sveshnikov", "e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 e5 Ndb5 d6"),
    ("ouverture:sicilienne-classique", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6"),
    ("ouverture:sicilienne-richter-rauzer", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bg5 e6"),
    ("ouverture:sicilienne-sozin", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 Nc6 Bc4 e6"),
    ("ouverture:sicilienne-scheveningue", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 e6 Be2 Be7"),
    ("ouverture:sicilienne-taimanov", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7"),
    ("ouverture:sicilienne-kan", "e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Nc3 Qc7"),
    ("ouverture:sicilienne-alapin", "e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6 cxd4 d6"),
    ("ouverture:sicilienne-alapin-d5", "e4 c5 c3 d5 exd5 Qxd5 d4 Nf6 Nf3 e6"),
    ("ouverture:sicilienne-fermee", "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6"),
    ("ouverture:sicilienne-rossolimo", "e4 c5 Nf3 Nc6 Bb5 g6 Bxc6 dxc6 d3 Bg7"),
    ("ouverture:sicilienne-moscou", "e4 c5 Nf3 d6 Bb5+ Bd7 Bxd7+ Qxd7 O-O Nf6"),
    ("ouverture:sicilienne-grand-prix", "e4 c5 Nc3 Nc6 f4 g6 Nf3 Bg7 Bb5 Nd4"),
    // --- Française ---
    ("ouverture:francaise-avance", "e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6"),
    ("ouverture:francaise-tarrasch", "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6"),
    ("ouverture:francaise-tarrasch-c5", "e4 e6 d4 d5 Nd2 c5 exd5 exd5 Ngf3 Nc6 Bb5 Bd6"),
    ("ouverture:francaise-winawer", "e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7"),
    ("ouverture:francaise-classique", "e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7"),
    ("ouverture:francaise-steinitz", "e4 e6 d4 d5 Nc3 Nf6 e5 Nfd7 f4 c5 Nf3 Nc6"),
    ("ouverture:francaise-rubinstein", "e4 e6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6"),
    ("ouverture:francaise-echange", "e4 e6 d4 d5 exd5 exd5 Nf3 Nf6 Bd3 Bd6 O-O O-O"),
    // --- Caro-Kann ---
    ("ouverture:caro-kann-classique", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6"),
    ("ouverture:caro-kann-nd7", "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Nf3 Ngf6 Nxf6+ Nxf6"),
    ("ouverture:caro-kann-avance", "e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5"),
    ("ouverture:caro-kann-avance-shirov", "e4 c6 d4 d5 e5 Bf5 Nc3 e6 g4 Bg6"),
    ("ouverture:caro-kann-panov", "e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6 Nf3 Bb4"),
    ("ouverture:caro-kann-echange", "e4 c6 d4 d5 exd5 cxd5 Bd3 Nc6 c3 Nf6 Bf4 Bg4"),
    ("ouverture:caro-kann-deux-cavaliers", "e4 c6 Nc3 d5 Nf3 Bg4 h3 Bxf3 Qxf3 e6"),
    ("ouverture:caro-kann-fantaisie", "e4 c6 d4 d5 f3 dxe4 fxe4 e5 Nf3 exd4"),
    // --- Gambit dame ---
    ("ouverture:gambit-dame-accepte", "d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6"),
    ("ouverture:gambit-dame-accepte-e4", "d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Nc6"),
    ("ouverture:gambit-dame-orthodoxe", "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7"),
    ("ouverture:gambit-dame-echange", "d4 d5 c4 e6 Nc3 Nf6 cxd5 exd5 Bg5 Be7 e3 c6"),
    ("ouverture:gambit-dame-tartakover", "d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bg5 h6 Bh4 O-O"),
    ("ouverture:gambit-dame-tarrasch", "d4 d5 c4 e6 Nc3 c5 cxd5 exd5 Nf3 Nc6 g3 Nf6"),
    ("ouverture:gambit-dame-semi-tarrasch", "d4 d5 c4 e6 Nc3 Nf6 Nf3 c5 cxd5 Nxd5"),
    ("ouverture:gambit-dame-chigorin", "d4 d5 c4 Nc6 Nf3 Bg4 cxd5 Bxf3 gxf3 Qxd5 e3 e5"),
    ("ouverture:contre-gambit-albin", "d4 d5 c4 e5 dxe5 d4 Nf3 Nc6"),
    // --- Slave et semi-slave ---
    ("ouverture:slave-principale", "d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6"),
    ("ouverture:slave-echange", "d4 d5 c4 c6 cxd5 cxd5 Nc3 Nf6 Nf3 Nc6 Bf4 Bf5"),
    ("ouverture:slave-chebanenko", "d4 d5 c4 c6 Nf3 Nf6 Nc3 a6 e3 b5"),
    ("ouverture:semi-slave-meran", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4"),
    ("ouverture:semi-slave-anti-meran", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 h6 Bxf6 Qxf6"),
    ("ouverture:semi-slave-botvinnik", "d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 Bg5 dxc4 e4 b5"),
    // --- Est-indienne ---
    ("ouverture:est-indienne-classique", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5"),
    ("ouverture:est-indienne-samisch", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5"),
    ("ouverture:est-indienne-averbakh", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Be2 O-O Bg5 c5"),
    ("ouverture:est-indienne-fianchetto", "d4 Nf6 c4 g6 Nf3 Bg7 g3 O-O Bg2 d6 O-O Nbd7"),
    ("ouverture:est-indienne-quatre-pions", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f4 O-O Nf3 c5"),
    // --- Nimzo-indienne ---
    ("ouverture:nimzo-indienne-rubinstein", "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5"),
    ("ouverture:nimzo-indienne-classique", "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6"),
    ("ouverture:nimzo-indienne-samisch", "d4 Nf6 c4 e6 Nc3 Bb4 a3 Bxc3+ bxc3 c5 f3 d5"),
    ("ouverture:nimzo-indienne-leningrad", "d4 Nf6 c4 e6 Nc3 Bb4 Bg5 h6 Bh4 c5 d5 d6"),
    // --- Ouest-indienne et Bogo ---
    ("ouverture:ouest-indienne", "d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O"),
    ("ouverture:ouest-indienne-petrosian", "d4 Nf6 c4 e6 Nf3 b6 a3 Bb7 Nc3 d5 cxd5 Nxd5"),
    ("ouverture:bogo-indienne", "d4 Nf6 c4 e6 Nf3 Bb4+ Bd2 Qe7 g3 Nc6"),
    // --- Grünfeld ---
    ("ouverture:grunfeld-echange", "d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7"),
    ("ouverture:grunfeld-russe", "d4 Nf6 c4 g6 Nc3 d5 Nf3 Bg7 Qb3 dxc4 Qxc4 O-O"),
    ("ouverture:grunfeld-bf4", "d4 Nf6 c4 g6 Nc3 d5 Bf4 Bg7 e3 O-O"),
    // --- Catalane ---
    ("ouverture:catalane-ouverte", "d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Nf3 a6 O-O Nc6"),
    ("ouverture:catalane-fermee", "d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O Nbd7"),
    // --- Anglaise ---
    ("ouverture:anglaise-symetrique", "c4 c5 Nf3 Nf6 Nc3 Nc6 g3 g6 Bg2 Bg7 O-O O-O"),
    ("ouverture:anglaise-quatre-cavaliers", "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6"),
    ("ouverture:anglaise-sicilienne-inversee", "c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5 Bg2 Nb6 Nf3 Nc6"),
    ("ouverture:anglaise-botvinnik", "c4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 e4 d6"),
    ("ouverture:anglaise-herisson", "c4 c5 Nf3 Nf6 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7"),
    ("ouverture:anglaise-mikenas", "c4 Nf6 Nc3 e6 e4 d5 e5 d4 exf6 dxc3 bxc3 Qxf6"),
    ("ouverture:anglaise-agincourt", "c4 e6 Nf3 d5 g3 Nf6 Bg2 Be7 O-O O-O"),
    // --- Réti ---
    ("ouverture:reti", "Nf3 d5 c4 e6 g3 Nf6 Bg2 Be7 O-O O-O"),
    ("ouverture:reti-slave", "Nf3 d5 c4 c6 g3 Nf6 Bg2 Bf5 O-O e6"),
    ("ouverture:attaque-est-indienne", "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5"),
    ("ouverture:reti-double-fianchetto", "Nf3 Nf6 g3 g6 b3 Bg7 Bb2 O-O Bg2 d6 O-O e5"),
    // --- Systèmes en d4 ---
    ("ouverture:london", "d4 d5 Nf3 Nf6 Bf4 c5 e3 Nc6 c3 e6 Nbd2 Bd6"),
    ("ouverture:london-indienne", "d4 Nf6 Nf3 g6 Bf4 Bg7 e3 O-O h3 d6"),
    ("ouverture:trompowsky", "d4 Nf6 Bg5 Ne4 Bf4 d5 e3 c5 Bd3 Nf6"),
    ("ouverture:trompowsky-e6", "d4 Nf6 Bg5 e6 e4 h6 Bxf6 Qxf6 Nc3 d6"),
    ("ouverture:jobava", "d4 Nf6 Nc3 d5 Bf4 e6 e3 Bd6 Bg3 O-O"),
    ("ouverture:colle", "d4 d5 Nf3 Nf6 e3 e6 Bd3 c5 c3 Nc6 Nbd2 Bd6"),
    ("ouverture:attaque-torre", "d4 Nf6 Nf3 e6 Bg5 c5 e3 h6 Bh4 b6"),
    ("ouverture:gambit-blackmar-diemer", "d4 d5 e4 dxe4 Nc3 Nf6 f3 exf3 Nxf3 g6"),
    // --- Écossaise ---
    ("ouverture:ecossaise-classique", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7"),
    ("ouverture:ecossaise-mieses", "e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7"),
    ("ouverture:gambit-ecossais", "e4 e5 Nf3 Nc6 d4 exd4 Bc4 Bc5 c3 Nf6 cxd4 Bb4+"),
    // --- Petrov ---
    ("ouverture:petrov-classique", "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Be7"),
    ("ouverture:petrov-moderne", "e4 e5 Nf3 Nf6 d4 Nxe4 Bd3 d5 Nxe5 Nd7"),
    // --- Philidor ---
    ("ouverture:philidor", "e4 e5 Nf3 d6 d4 exd4 Nxd4 Nf6 Nc3 Be7"),
    ("ouverture:philidor-moderne", "e4 d6 d4 Nf6 Nc3 e5 Nf3 Nbd7 Bc4 Be7"),
    // --- Pirc et moderne ---
    ("ouverture:pirc-classique", "e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6"),
    ("ouverture:pirc-autrichienne", "e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O Bd3 Na6"),
    ("ouverture:pirc-150", "e4 d6 d4 Nf6 Nc3 g6 Be3 Bg7 Qd2 c6 f3 b5"),
    ("ouverture:moderne", "e4 g6 d4 Bg7 Nc3 d6 Be3 a6 Qd2 b5"),
    // --- Scandinave ---
    ("ouverture:scandinave-qa5", "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5"),
    ("ouverture:scandinave-qd6", "e4 d5 exd5 Qxd5 Nc3 Qd6 d4 Nf6 Nf3 a6"),
    ("ouverture:scandinave-nf6", "e4 d5 exd5 Nf6 d4 Nxd5 Nf3 g6 Be2 Bg7"),
    // --- Alekhine (la vraie ligne 2.e5 Nd5) ---
    ("ouverture:alekhine-quatre-pions", "e4 Nf6 e5 Nd5 d4 d6 c4 Nb6 f4 dxe5 fxe5 Nc6"),
    ("ouverture:alekhine-moderne", "e4 Nf6 e5 Nd5 d4 d6 Nf3 g6 Bc4 Nb6 Bb3 Bg7"),
    ("ouverture:alekhine-echange", "e4 Nf6 e5 Nd5 d4 d6 c4 Nb6 exd6 cxd6"),
    // --- Benoni, Benko, Budapest ---
    ("ouverture:benoni-moderne", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 e4 g6"),
    ("ouverture:benoni-fianchetto", "d4 Nf6 c4 c5 d5 e6 Nc3 exd5 cxd5 d6 Nf3 g6"),
    ("ouverture:gambit-benko", "d4 Nf6 c4 c5 d5 b5 cxb5 a6 bxa6 Bxa6 Nc3 d6"),
    ("ouverture:gambit-budapest", "d4 Nf6 c4 e5 dxe5 Ng4 Bf4 Nc6 Nf3 Bb4+ Nbd2 Qe7"),
    // --- Hollandaise ---
    ("ouverture:hollandaise-leningrad", "d4 f5 g3 Nf6 Bg2 g6 Nf3 Bg7 O-O O-O c4 d6"),
    ("ouverture:hollandaise-stonewall", "d4 f5 g3 Nf6 Bg2 e6 Nf3 d5 O-O Bd6 c4 c6"),
    ("ouverture:hollandaise-classique", "d4 f5 c4 Nf6 g3 e6 Bg2 Be7 Nf3 O-O O-O d6"),
    ("ouverture:hollandaise-gambit-staunton", "d4 f5 e4 fxe4 Nc3 Nf6 Bg5 Nc6"),
    // --- Gambit du roi ---
    ("ouverture:gambit-roi-kieseritzky", "e4 e5 f4 exf4 Nf3 g5 h4 g4 Ne5 Nf6"),
    ("ouverture:gambit-roi-refuse", "e4 e5 f4 Bc5 Nf3 d6 Nc3 Nf6 Bc4 Nc6 d3 a6"),
    ("ouverture:gambit-roi-moderne", "e4 e5 f4 exf4 Nf3 d5 exd5 Nf6 Bb5+ c6"),
    // --- Vienne ---
    ("ouverture:vienne-gambit", "e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7"),
    ("ouverture:vienne-fianchetto", "e4 e5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6"),
    // --- Autres débuts ouverts et divers ---
    ("ouverture:partie-du-centre", "e4 e5 d4 exd4 Qxd4 Nc6 Qe3 Nf6 Nc3 Bb4 Bd2 O-O"),
    ("ouverture:ponziani", "e4 e5 Nf3 Nc6 c3 Nf6 d4 Nxe4 d5 Ne7 Nxe5 Ng6"),
    ("ouverture:quatre-cavaliers", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4 O-O O-O d3 d6"),
    ("ouverture:quatre-cavaliers-ecossais", "e4 e5 Nf3 Nc6 Nc3 Nf6 d4 exd4 Nxd4 Bb4 Nxc6 bxc6"),
    ("ouverture:partie-du-fou", "e4 e5 Bc4 Nf6 d3 c6 Nf3 d5 Bb3 Bd6"),
    ("ouverture:bird", "f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5"),
    ("ouverture:larsen", "b3 e5 Bb2 Nc6 e3 Nf6 Bb5 Bd6 Nf3 Qe7"),
    ("ouverture:owen", "e4 b6 d4 Bb7 Bd3 e6 Nf3 c5 c3 Nf6 Qe2 Be7"),
    ("ouverture:nimzowitsch", "e4 Nc6 d4 d5 e5 Bf5 c3 e6"),
    // === FIN DU LIVRE ===
];

/// Livre parsé : positions finales des lignes (pour `tirage`) et continuations
/// indexées par zobrist de chaque position-préfixe (pour `coup_du_livre`).
struct Livre {
    /// (étiquette, position après le dernier coup de la ligne).
    finales_de_lignes: Vec<(&'static str, Chess)>,
    /// zobrist(position) → coups de continuation, UN par ligne passant par la
    /// position (les coups populaires pèsent donc naturellement plus lourd).
    continuations: HashMap<u64, Vec<Move>>,
}

static LIVRE: OnceLock<Livre> = OnceLock::new();

/// Parse tout le livre au premier accès. Une ligne illégale = panic immédiat
/// avec l'étiquette ET le coup fautif (le test `livre_entier_parse` l'attrape).
fn livre() -> &'static Livre {
    LIVRE.get_or_init(|| {
        let mut finales_de_lignes = Vec::with_capacity(LIGNES.len());
        let mut continuations: HashMap<u64, Vec<Move>> = HashMap::new();
        for (etiquette, sans) in LIGNES {
            let mut pos = Chess::default();
            for san in sans.split_whitespace() {
                let coup = san
                    .parse::<SanPlus>()
                    .unwrap_or_else(|e| {
                        panic!("livre d'ouvertures, ligne « {etiquette} » : \
                                SAN imparsable « {san} » ({e})")
                    })
                    .san
                    .to_move(&pos)
                    .unwrap_or_else(|e| {
                        panic!("livre d'ouvertures, ligne « {etiquette} » : \
                                coup illégal « {san} » ({e})")
                    });
                continuations.entry(zobrist(&pos)).or_default().push(coup.clone());
                pos = pos.play(&coup).unwrap_or_else(|e| {
                    panic!("livre d'ouvertures, ligne « {etiquette} » : \
                            coup injouable « {san} » ({e})")
                });
            }
            finales_de_lignes.push((*etiquette, pos));
        }
        Livre { finales_de_lignes, continuations }
    })
}

/// Coup du livre pour le PLATEAU : si `pos` est un préfixe d'une ou plusieurs
/// lignes du livre (comparaison par zobrist, transpositions comprises),
/// renvoie un coup de continuation tiré au hasard parmi elles ; None sinon.
/// Chaque coup candidat est revérifié légal dans `pos` (blindage contre une
/// collision zobrist — théorique, mais un coup illégal ferait paniquer serve).
pub fn coup_du_livre(pos: &Chess, rng: &mut StdRng) -> Option<Move> {
    let candidats = livre().continuations.get(&zobrist(pos))?;
    let legaux: Vec<&Move> = candidats
        .iter()
        .filter(|m| pos.legal_moves().contains(*m))
        .collect();
    legaux.choose(rng).map(|m| (*m).clone())
}

// ---------------------------------------------------------------------------
// Générateur de finales
// ---------------------------------------------------------------------------

/// Gabarits de matériel : (étiquette, pièces blanches HORS roi, pièces noires
/// HORS roi). Les rois sont implicites. Mélange de finales de pions, de tours,
/// de dames et de pièces mineures, plus les mats élémentaires (KQvK, KRvK,
/// KBNvK) que le self-play depuis la position initiale ne voit presque jamais.
const GABARITS: &[(&str, &[Role], &[Role])] = &[
    ("finale:KPvK", &[Role::Pawn], &[]),
    ("finale:KPPvK", &[Role::Pawn, Role::Pawn], &[]),
    ("finale:KPPvKP", &[Role::Pawn, Role::Pawn], &[Role::Pawn]),
    ("finale:KPPPvKPP", &[Role::Pawn, Role::Pawn, Role::Pawn], &[Role::Pawn, Role::Pawn]),
    ("finale:KRvKR", &[Role::Rook], &[Role::Rook]),
    ("finale:KRPvKR", &[Role::Rook, Role::Pawn], &[Role::Rook]),
    ("finale:KRPPvKRP", &[Role::Rook, Role::Pawn, Role::Pawn], &[Role::Rook, Role::Pawn]),
    ("finale:KQvKQ", &[Role::Queen], &[Role::Queen]),
    ("finale:KQPvKQ", &[Role::Queen, Role::Pawn], &[Role::Queen]),
    ("finale:KBPvKB", &[Role::Bishop, Role::Pawn], &[Role::Bishop]),
    ("finale:KNPvKN", &[Role::Knight, Role::Pawn], &[Role::Knight]),
    ("finale:KBPvKN", &[Role::Bishop, Role::Pawn], &[Role::Knight]),
    ("finale:KRvKP", &[Role::Rook], &[Role::Pawn]),
    ("finale:KQvKR", &[Role::Queen], &[Role::Rook]),
    ("finale:KQvK", &[Role::Queen], &[]),
    ("finale:KRvK", &[Role::Rook], &[]),
    ("finale:KBNvK", &[Role::Bishop, Role::Knight], &[]),
];

/// Case libre au hasard : pions confinés aux rangées 2 à 7, tout le reste
/// n'importe où. Retente tant que la case est occupée.
fn case_libre(board: &Board, role: Role, rng: &mut StdRng) -> Square {
    loop {
        let file = File::new(rng.gen_range(0..8));
        let rank = if role == Role::Pawn {
            Rank::new(rng.gen_range(1..7)) // jamais les rangées 1 et 8
        } else {
            Rank::new(rng.gen_range(0..8))
        };
        let sq = Square::from_coords(file, rank);
        if board.piece_at(sq).is_none() {
            return sq;
        }
    }
}

/// Génère une finale légale depuis un gabarit tiré au hasard : rois non
/// adjacents, pions hors des rangées 1/8, côté au trait aléatoire ; la
/// validation finale est déléguée à shakmaty (`Setup` → `Chess`), qui rejette
/// notamment un camp SANS le trait en échec — on retente alors. On écarte
/// aussi les positions sans aucun coup légal (mat ou pat immédiat : rien à
/// apprendre d'une partie déjà finie).
fn genere_finale(rng: &mut StdRng) -> (Chess, &'static str) {
    let (etiquette, blancs, noirs) = *GABARITS.choose(rng).expect("gabarits non vides");
    // Bien plus qu'il n'en faut : chaque tentative réussit avec une forte
    // probabilité, la boucle en consomme une poignée en pratique.
    for _ in 0..100_000 {
        let mut board = Board::empty();
        // Rois d'abord : non adjacents (distance de Tchebychev >= 2).
        let roi_blanc = case_libre(&board, Role::King, rng);
        board.set_piece_at(roi_blanc, Role::King.of(Color::White));
        let roi_noir = case_libre(&board, Role::King, rng);
        if roi_blanc.distance(roi_noir) < 2 {
            continue;
        }
        board.set_piece_at(roi_noir, Role::King.of(Color::Black));
        // Puis le matériel du gabarit, sur des cases libres.
        for &role in blancs {
            let sq = case_libre(&board, role, rng);
            board.set_piece_at(sq, role.of(Color::White));
        }
        for &role in noirs {
            let sq = case_libre(&board, role, rng);
            board.set_piece_at(sq, role.of(Color::Black));
        }
        let setup = Setup {
            board,
            turn: if rng.gen::<bool>() { Color::White } else { Color::Black },
            ..Setup::empty()
        };
        // shakmaty valide tout le reste (camp sans le trait en échec, etc.).
        let Ok(pos) = setup.position::<Chess>(CastlingMode::Standard) else {
            continue;
        };
        if pos.legal_moves().is_empty() {
            continue;
        }
        return (pos, etiquette);
    }
    unreachable!("générateur de finales : aucune position légale en 100 000 essais")
}

// ---------------------------------------------------------------------------
// Générateur de milieux tardifs (transition fin-de-milieu → finale)
// ---------------------------------------------------------------------------

/// Gabarits de matériel des MILIEUX TARDIFS : (étiquette, pièces blanches
/// HORS roi, pièces noires HORS roi). 10 à 16 pièces au total (rois compris),
/// écart matériel ≤ 1 pion (barème classique P1 N3 B3 R5 Q9) — garanti par
/// construction, revérifié par le test. Variété : couples de fous/cavaliers,
/// tours doublées, dames présentes ou non, totaux pairs ET impairs.
const GABARITS_TRANSITION: &[(&str, &[Role], &[Role])] = &[
    // 10 pièces
    ("transition:KRNPPvKRNPP", &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn]),
    ("transition:KRBPPvKRNPP", &[Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn]),
    ("transition:KQPPPvKQPPP", &[Role::Queen, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KQNPPvKQBPP", &[Role::Queen, Role::Knight, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Bishop, Role::Pawn, Role::Pawn]),
    // 11 pièces (écart d'un pion)
    ("transition:KRRPPvKRRPPP", &[Role::Rook, Role::Rook, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Rook, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRNPPPvKRNPP",
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn]),
    ("transition:KQRPPPvKQRPP",
     &[Role::Queen, Role::Rook, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Rook, Role::Pawn, Role::Pawn]),
    // 12 pièces
    ("transition:KRNPPPvKRNPPP",
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRBPPPvKRNPPP",
     &[Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRRPPPvKRRPPP",
     &[Role::Rook, Role::Rook, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Rook, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KQRPPPvKQRPPP",
     &[Role::Queen, Role::Rook, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Rook, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRBNPPvKRBNPP",
     &[Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn]),
    ("transition:KRRNPPvKRRBPP",
     &[Role::Rook, Role::Rook, Role::Knight, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn]),
    ("transition:KQBPPPvKQNPPP",
     &[Role::Queen, Role::Bishop, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KQRBPPvKQRNPP",
     &[Role::Queen, Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Rook, Role::Knight, Role::Pawn, Role::Pawn]),
    // 13 pièces (écart d'un pion)
    ("transition:KRNPPPPvKRBPPP",
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn, Role::Pawn]),
    // 14 pièces
    ("transition:KBNPPPPvKBNPPPP",
     &[Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRBPPPPvKRNPPPP",
     &[Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRNNPPPvKRBBPPP",
     &[Role::Rook, Role::Knight, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Bishop, Role::Bishop, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KQBNPPPvKQBNPPP",
     &[Role::Queen, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn]),
    ("transition:KRRBNPPvKRRBNPP",
     &[Role::Rook, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn]),
    ("transition:KQRBNPPvKQRBNPP",
     &[Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn]),
    // 15 pièces (écart d'un pion)
    ("transition:KQRBPPPvKQRNPPPP",
     &[Role::Queen, Role::Rook, Role::Bishop, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Queen, Role::Rook, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn,
       Role::Pawn]),
    ("transition:KRBNPPPvKRBNPPPP",
     &[Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn],
     &[Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn,
       Role::Pawn]),
    // 16 pièces
    ("transition:KRBNPPPPvKRBNPPPP",
     &[Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn,
       Role::Pawn],
     &[Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn, Role::Pawn,
       Role::Pawn]),
    ("transition:KQRBNPPPvKQRBNPPP",
     &[Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn,
       Role::Pawn],
     &[Role::Queen, Role::Rook, Role::Bishop, Role::Knight, Role::Pawn, Role::Pawn,
       Role::Pawn]),
];

/// Prise « gagnante évidente » : SEE ≥ un pion entier (centièmes de pion,
/// barème de search::valeur_see).
const SEUIL_PRISE_GAGNANTE: i32 = 100;

/// Bande de rangées plausible (indices 0..=7, bornes incluses) d'une pièce
/// d'un milieu tardif : rois abrités ou semi-exposés sur leurs deux premières
/// rangées, pions ni sur leur rangée de promotion ni sur la pré-dernière
/// (aucune promotion imminente qui fausserait l'équilibre matériel affiché),
/// le reste n'importe où.
fn bande_transition(role: Role, couleur: Color) -> (u32, u32) {
    match (role, couleur) {
        (Role::King, Color::White) => (0, 1), // rangées 1-2
        (Role::King, Color::Black) => (6, 7), // rangées 7-8
        (Role::Pawn, Color::White) => (1, 5), // rangées 2-6
        (Role::Pawn, Color::Black) => (2, 6), // rangées 3-7
        _ => (0, 7),
    }
}

/// Case libre au hasard dans une bande de rangées [min, max] (bornes
/// incluses). Retente tant que la case est occupée.
fn case_libre_bande(board: &Board, min: u32, max: u32, rng: &mut StdRng) -> Square {
    loop {
        let file = File::new(rng.gen_range(0..8));
        let rank = Rank::new(rng.gen_range(min..=max));
        let sq = Square::from_coords(file, rank);
        if board.piece_at(sq).is_none() {
            return sq;
        }
    }
}

/// Génère un MILIEU TARDIF légal et calme depuis un gabarit tiré au hasard :
/// rois sur leurs deux premières rangées (jamais adjacents, les bandes sont
/// disjointes), pions hors des rangées extrêmes, côté au trait aléatoire.
/// Filtres de plausibilité, dans l'ordre du moins cher au plus cher :
/// - shakmaty valide la position (`Setup` → `Chess`) ;
/// - il reste des coups légaux (ni mat ni pat : rien à apprendre sinon) ;
/// - le camp au trait n'est PAS en échec (départ calme) ;
/// - recherche éclair de vérification : AUCUNE prise du camp au trait ne
///   gagne un pion ou plus à l'échange (`search::see` sur chaque prise
///   légale — l'échange optimal complet, rayons X compris, pas seulement la
///   victime immédiate). Une pièce « en prise triviale » est ainsi écartée.
fn genere_transition(rng: &mut StdRng) -> (Chess, &'static str) {
    let (etiquette, blancs, noirs) =
        *GABARITS_TRANSITION.choose(rng).expect("gabarits transition non vides");
    // Les filtres rejettent la grosse majorité des placements bruts (prises
    // gagnantes surtout) : quelques dizaines d'essais suffisent en pratique,
    // la borne est très au-dessus.
    for _ in 0..100_000 {
        let mut board = Board::empty();
        for (couleur, roles) in [(Color::White, blancs), (Color::Black, noirs)] {
            let (min, max) = bande_transition(Role::King, couleur);
            let roi = case_libre_bande(&board, min, max, rng);
            board.set_piece_at(roi, Role::King.of(couleur));
            for &role in roles {
                let (min, max) = bande_transition(role, couleur);
                let sq = case_libre_bande(&board, min, max, rng);
                board.set_piece_at(sq, role.of(couleur));
            }
        }
        let setup = Setup {
            board,
            turn: if rng.gen::<bool>() { Color::White } else { Color::Black },
            ..Setup::empty()
        };
        let Ok(pos) = setup.position::<Chess>(CastlingMode::Standard) else {
            continue;
        };
        if pos.legal_moves().is_empty() || pos.is_check() {
            continue;
        }
        if pos
            .capture_moves()
            .iter()
            .any(|m| crate::search::see(&pos, m) >= SEUIL_PRISE_GAGNANTE)
        {
            continue;
        }
        return (pos, etiquette);
    }
    unreachable!("générateur de transition : aucune position légale en 100 000 essais")
}

// ---------------------------------------------------------------------------
// Tirage
// ---------------------------------------------------------------------------

/// Valide un jeu de parts destiné à `tirage_complet` : chaque part dans
/// [0, 1] et somme ≤ 1 (tolérance d'arrondi f32 pour une somme voulue
/// exactement à 1). Refus explicite en amont plutôt que la troncature
/// silencieuse du tirage : des seuils cumulés > 1 écraseraient la dernière
/// famille et feraient disparaître la famille « initiale » sans un mot.
/// Utilisée par train.exe et calibration.exe au parsing des options.
pub fn valide_parts(
    p_ouverture: f32,
    p_finale: f32,
    p_transition: f32,
) -> Result<(), String> {
    for (nom, p) in [
        ("ouvertures", p_ouverture),
        ("finales", p_finale),
        ("transition", p_transition),
    ] {
        if !(0.0..=1.0).contains(&p) {
            return Err(format!(
                "part de départs « {nom} » invalide : {p} (attendue dans [0, 1])"
            ));
        }
    }
    let somme = p_ouverture + p_finale + p_transition;
    if somme > 1.0 + 1e-6 {
        return Err(format!(
            "parts de départs incohérentes : ouvertures {p_ouverture} + \
             finales {p_finale} + transition {p_transition} = {somme} > 1"
        ));
    }
    Ok(())
}

/// Tire une position de départ : ouverture du livre avec probabilité
/// `p_ouverture`, finale générée avec probabilité `p_finale`, position
/// initiale sinon. Les plis chauds suivent la famille : 8 (initiale,
/// comportement historique), 2 (ouverture), 0 (finale).
/// Signature historique INTACTE : délègue à `tirage_complet` avec une part
/// de transition NULLE (mêmes consommations du rng, mêmes tirages qu'avant).
pub fn tirage(rng: &mut StdRng, p_ouverture: f32, p_finale: f32) -> Depart {
    tirage_complet(rng, p_ouverture, p_finale, 0.0)
}

/// Tirage à QUATRE familles : ouverture du livre (`p_ouverture`), finale
/// générée (`p_finale`), milieu tardif généré (`p_transition`), position
/// initiale sinon. Plis chauds : 8 (initiale), 2 (ouverture), 0 (finale et
/// transition — la position générée est déjà diversifiée, autant y jouer
/// précis dès le premier coup). `p_transition` = 0 → strictement `tirage`.
pub fn tirage_complet(
    rng: &mut StdRng,
    p_ouverture: f32,
    p_finale: f32,
    p_transition: f32,
) -> Depart {
    let u: f32 = rng.gen();
    if u < p_ouverture {
        let (etiquette, pos) = livre()
            .finales_de_lignes
            .choose(rng)
            .expect("livre non vide");
        Depart { pos: pos.clone(), etiquette, plis_chauds: 2 }
    } else if u < p_ouverture + p_finale {
        let (pos, etiquette) = genere_finale(rng);
        Depart { pos, etiquette, plis_chauds: 0 }
    } else if u < p_ouverture + p_finale + p_transition {
        let (pos, etiquette) = genere_transition(rng);
        Depart { pos, etiquette, plis_chauds: 0 }
    } else {
        Depart { pos: Chess::default(), etiquette: "initiale", plis_chauds: 8 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use shakmaty::san::San;

    /// TOUTES les lignes du livre parsent (itère TOUT le livre, coup par
    /// coup) ; au moins 120 lignes, 6 à 12 demi-coups chacune, étiquettes
    /// « ouverture:... » ; le chargement OnceLock voit le même compte.
    #[test]
    fn livre_entier_parse() {
        assert!(
            LIGNES.len() >= 120,
            "livre trop maigre : {} lignes (minimum 120)",
            LIGNES.len()
        );
        for (etiquette, sans) in LIGNES {
            assert!(
                etiquette.starts_with("ouverture:"),
                "étiquette sans préfixe ouverture: « {etiquette} »"
            );
            let n = sans.split_whitespace().count();
            assert!(
                (6..=12).contains(&n),
                "ligne « {etiquette} » : {n} demi-coups (attendu 6 à 12)"
            );
            // Re-parse indépendant du chargeur : le message d'échec nomme la
            // ligne ET le coup fautif.
            let mut pos = Chess::default();
            for san in sans.split_whitespace() {
                let coup = san
                    .parse::<SanPlus>()
                    .unwrap_or_else(|e| panic!("« {etiquette} » : SAN « {san} » ({e})"))
                    .san
                    .to_move(&pos)
                    .unwrap_or_else(|e| panic!("« {etiquette} » : coup « {san} » ({e})"));
                pos = pos.play(&coup).expect("coup légal donc jouable");
            }
        }
        // Le chargeur (qui panique sur une ligne illégale) voit tout le livre.
        assert_eq!(livre().finales_de_lignes.len(), LIGNES.len());
    }

    /// 500 finales générées : toutes légales (revalidation Setup → Chess),
    /// rois non adjacents, aucun pion sur les rangées 1/8, étiquette
    /// « finale:... » cohérente avec un gabarit.
    #[test]
    fn finales_500_legales() {
        let mut rng = StdRng::seed_from_u64(42);
        for i in 0..500 {
            let (pos, etiquette) = genere_finale(&mut rng);
            assert!(etiquette.starts_with("finale:"), "étiquette « {etiquette} »");
            let setup = pos.clone().into_setup(EnPassantMode::Legal);
            // Rois non adjacents.
            let rois = setup.board.kings();
            let (a, b) = (
                rois.first().expect("roi blanc présent"),
                rois.last().expect("roi noir présent"),
            );
            assert!(a.distance(b) >= 2, "finale {i} ({etiquette}) : rois adjacents");
            // Pions hors des rangées 1 et 8.
            for sq in setup.board.pawns() {
                assert!(
                    sq.rank() != Rank::First && sq.rank() != Rank::Eighth,
                    "finale {i} ({etiquette}) : pion en {sq}"
                );
            }
            // Revalidation complète par shakmaty.
            assert!(
                setup.position::<Chess>(CastlingMode::Standard).is_ok(),
                "finale {i} ({etiquette}) : position invalide"
            );
            assert!(!pos.legal_moves().is_empty(), "finale {i} : partie déjà finie");
        }
    }

    /// Valeur matérielle en pions entiers (barème classique P1 N3 B3 R5 Q9)
    /// d'un camp — pour vérifier l'équilibre des milieux tardifs générés.
    fn valeur_camp(board: &Board, couleur: Color) -> i32 {
        [
            (Role::Pawn, 1),
            (Role::Knight, 3),
            (Role::Bishop, 3),
            (Role::Rook, 5),
            (Role::Queen, 9),
        ]
        .iter()
        .map(|&(role, v)| (board.by_color(couleur) & board.by_role(role)).count() as i32 * v)
        .sum()
    }

    /// 200 milieux tardifs générés : tous légaux (revalidation Setup → Chess),
    /// 10 à 16 pièces au total, écart matériel ≤ 1 pion, rois dans leurs
    /// bandes (blanc rangées 1-2, noir 7-8), pions hors des rangées 1/8,
    /// camp au trait ni en échec ni devant une prise gagnante évidente
    /// (SEE < 1 pion sur TOUTES ses prises légales), étiquette
    /// « transition:... » ; la distribution du compte de pièces couvre la
    /// plage (au moins 5 totaux distincts) et est affichée (--nocapture).
    #[test]
    fn transition_200_legales() {
        let mut rng = StdRng::seed_from_u64(9);
        let mut par_compte: HashMap<u32, u32> = HashMap::new();
        for i in 0..200 {
            let (pos, etiquette) = genere_transition(&mut rng);
            assert!(etiquette.starts_with("transition:"), "étiquette « {etiquette} »");
            let setup = pos.clone().into_setup(EnPassantMode::Legal);
            // Compte de pièces total (rois compris) dans la plage 10-16.
            let n = setup.board.occupied().count() as u32;
            assert!((10..=16).contains(&n), "transition {i} ({etiquette}) : {n} pièces");
            *par_compte.entry(n).or_default() += 1;
            // Écart matériel ≤ 1 pion.
            let ecart =
                (valeur_camp(&setup.board, Color::White) - valeur_camp(&setup.board, Color::Black))
                    .abs();
            assert!(ecart <= 1, "transition {i} ({etiquette}) : écart {ecart} pions");
            // Rois abrités dans leurs bandes.
            let roi_blanc = setup.board.king_of(Color::White).expect("roi blanc");
            let roi_noir = setup.board.king_of(Color::Black).expect("roi noir");
            assert!(u32::from(roi_blanc.rank()) <= 1, "transition {i} : roi blanc {roi_blanc}");
            assert!(u32::from(roi_noir.rank()) >= 6, "transition {i} : roi noir {roi_noir}");
            // Pions hors des rangées 1 et 8.
            for sq in setup.board.pawns() {
                assert!(
                    sq.rank() != Rank::First && sq.rank() != Rank::Eighth,
                    "transition {i} ({etiquette}) : pion en {sq}"
                );
            }
            // Départ calme : pas en échec, des coups légaux, et AUCUNE prise
            // gagnante évidente au premier coup (recherche éclair SEE).
            assert!(!pos.is_check(), "transition {i} ({etiquette}) : au trait en échec");
            assert!(!pos.legal_moves().is_empty(), "transition {i} : partie déjà finie");
            for m in pos.capture_moves() {
                let g = crate::search::see(&pos, &m);
                assert!(
                    g < SEUIL_PRISE_GAGNANTE,
                    "transition {i} ({etiquette}) : prise gagnante {m:?} (SEE {g})"
                );
            }
            // Revalidation complète par shakmaty.
            assert!(
                setup.position::<Chess>(CastlingMode::Standard).is_ok(),
                "transition {i} ({etiquette}) : position invalide"
            );
        }
        // Variété de la distribution : au moins 5 totaux distincts sur 200.
        assert!(par_compte.len() >= 5, "distribution étroite : {par_compte:?}");
        let mut comptes: Vec<_> = par_compte.into_iter().collect();
        comptes.sort();
        println!("distribution du compte de pièces (200 transitions) : {comptes:?}");
    }

    /// COUPERET TRANSITION (harnais opérateur, ignoré par défaut) : 500
    /// milieux tardifs générés, chacun vérifié contre TOUTES les clauses du
    /// contrat — légalité stricte (revalidation Setup → Chess), 10 à 16
    /// pièces, écart matériel ≤ 1 pion, rois dans leurs bandes, pions hors
    /// des rangées extrêmes, départ calme (pas d'échec, pas de prise
    /// gagnante évidente au SEE), étiquette « transition:... ». La variété
    /// est mesurée et affichée : distribution des comptes de pièces (les 7
    /// totaux 10..16 attendus sur 500) et nombre de gabarits distincts.
    /// Lancer :
    /// cargo test --lib departs::tests::couperet_transition_500 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn couperet_transition_500() {
        let mut rng = StdRng::seed_from_u64(2026);
        let mut par_compte: HashMap<u32, u32> = HashMap::new();
        let mut gabarits: HashMap<&'static str, u32> = HashMap::new();
        for i in 0..500 {
            let (pos, etiquette) = genere_transition(&mut rng);
            assert!(etiquette.starts_with("transition:"), "étiquette « {etiquette} »");
            *gabarits.entry(etiquette).or_default() += 1;
            let setup = pos.clone().into_setup(EnPassantMode::Legal);
            let n = setup.board.occupied().count() as u32;
            assert!((10..=16).contains(&n), "transition {i} ({etiquette}) : {n} pièces");
            *par_compte.entry(n).or_default() += 1;
            let ecart =
                (valeur_camp(&setup.board, Color::White) - valeur_camp(&setup.board, Color::Black))
                    .abs();
            assert!(ecart <= 1, "transition {i} ({etiquette}) : écart {ecart} pions");
            let roi_blanc = setup.board.king_of(Color::White).expect("roi blanc");
            let roi_noir = setup.board.king_of(Color::Black).expect("roi noir");
            assert!(u32::from(roi_blanc.rank()) <= 1, "transition {i} : roi blanc {roi_blanc}");
            assert!(u32::from(roi_noir.rank()) >= 6, "transition {i} : roi noir {roi_noir}");
            for sq in setup.board.pawns() {
                assert!(
                    sq.rank() != Rank::First && sq.rank() != Rank::Eighth,
                    "transition {i} ({etiquette}) : pion en {sq}"
                );
            }
            assert!(!pos.is_check(), "transition {i} ({etiquette}) : au trait en échec");
            assert!(!pos.legal_moves().is_empty(), "transition {i} : partie déjà finie");
            for m in pos.capture_moves() {
                let g = crate::search::see(&pos, &m);
                assert!(
                    g < SEUIL_PRISE_GAGNANTE,
                    "transition {i} ({etiquette}) : prise gagnante {m:?} (SEE {g})"
                );
            }
            assert!(
                setup.position::<Chess>(CastlingMode::Standard).is_ok(),
                "transition {i} ({etiquette}) : position invalide"
            );
        }
        // Variété : les 7 comptes de la plage présents, et une majorité des
        // gabarits visités sur 500 tirages.
        assert!(par_compte.len() >= 5, "distribution étroite : {par_compte:?}");
        assert!(
            gabarits.len() >= GABARITS_TRANSITION.len() / 2,
            "gabarits visités : {} / {}",
            gabarits.len(),
            GABARITS_TRANSITION.len()
        );
        let mut comptes: Vec<_> = par_compte.into_iter().collect();
        comptes.sort();
        println!("couperet transition : 500/500 positions conformes");
        println!("distribution du compte de pièces : {comptes:?}");
        println!(
            "gabarits distincts visités : {} / {}",
            gabarits.len(),
            GABARITS_TRANSITION.len()
        );
    }

    /// tirage_complet respecte les quatre parts sur 2000 tirages
    /// (0.3/0.2/0.2 → ~600/400/400/600, tolérance large ±120), avec les plis
    /// chauds de chaque famille (2/0/0/8) ; et une part de transition NULLE
    /// reproduit `tirage` à l'identique (mêmes graines → mêmes départs).
    #[test]
    fn tirage_complet_quatre_familles() {
        let mut rng = StdRng::seed_from_u64(77);
        let (mut ouvertures, mut finales, mut transitions, mut initiales) = (0i32, 0i32, 0i32, 0i32);
        for _ in 0..2000 {
            let d = tirage_complet(&mut rng, 0.3, 0.2, 0.2);
            if d.etiquette.starts_with("ouverture:") {
                assert_eq!(d.plis_chauds, 2);
                ouvertures += 1;
            } else if d.etiquette.starts_with("finale:") {
                assert_eq!(d.plis_chauds, 0);
                finales += 1;
            } else if d.etiquette.starts_with("transition:") {
                assert_eq!(d.plis_chauds, 0);
                transitions += 1;
            } else {
                assert_eq!(d.etiquette, "initiale");
                assert_eq!(d.plis_chauds, 8);
                assert_eq!(d.pos, Chess::default());
                initiales += 1;
            }
        }
        assert!((ouvertures - 600).abs() <= 120, "ouvertures : {ouvertures}");
        assert!((finales - 400).abs() <= 120, "finales : {finales}");
        assert!((transitions - 400).abs() <= 120, "transitions : {transitions}");
        assert!((initiales - 600).abs() <= 120, "initiales : {initiales}");
        // Part de transition nulle = tirage historique, départ pour départ.
        let mut a = StdRng::seed_from_u64(4242);
        let mut b = StdRng::seed_from_u64(4242);
        for _ in 0..200 {
            let da = tirage(&mut a, 0.5, 0.25);
            let db = tirage_complet(&mut b, 0.5, 0.25, 0.0);
            assert_eq!(da.etiquette, db.etiquette);
            assert_eq!(da.pos, db.pos);
            assert_eq!(da.plis_chauds, db.plis_chauds);
        }
    }

    /// valide_parts accepte les réglages légitimes (dont le réglage de
    /// production 0.5/0.2/0.2 et une somme exactement à 1) et refuse part
    /// hors [0, 1], somme > 1 et NaN — le tirage ne doit jamais tronquer.
    #[test]
    fn valide_parts_refuse_les_parts_incoherentes() {
        assert!(valide_parts(0.0, 0.0, 0.0).is_ok());
        assert!(valide_parts(0.5, 0.2, 0.2).is_ok()); // réglage de production
        assert!(valide_parts(0.5, 0.25, 0.25).is_ok()); // somme exactement 1
        assert!(valide_parts(0.3, 0.3, 0.4).is_ok()); // somme 1 aux arrondis f32 près
        assert!(valide_parts(0.5, 0.4, 0.4).is_err()); // somme 1.3 : troncature
        assert!(valide_parts(-0.1, 0.2, 0.2).is_err()); // part négative
        assert!(valide_parts(0.2, 1.5, 0.0).is_err()); // part > 1
        assert!(valide_parts(0.2, f32::NAN, 0.2).is_err()); // NaN
    }

    /// Sur la position initiale, coup_du_livre renvoie un coup qui est bien le
    /// PREMIER coup d'au moins une ligne ; et après 1.e4 (préfixe de
    /// nombreuses lignes), il renvoie encore un coup.
    #[test]
    fn coup_du_livre_position_initiale() {
        let mut rng = StdRng::seed_from_u64(7);
        let pos = Chess::default();
        let coup = coup_du_livre(&pos, &mut rng).expect("position initiale au livre");
        let premiers: Vec<Move> = LIGNES
            .iter()
            .map(|(_, sans)| {
                let premier = sans.split_whitespace().next().expect("ligne non vide");
                premier
                    .parse::<SanPlus>()
                    .expect("SAN lisible")
                    .san
                    .to_move(&pos)
                    .expect("premier coup légal")
            })
            .collect();
        assert!(
            premiers.contains(&coup),
            "coup {coup:?} absent des premiers coups du livre"
        );
        // Un cran plus loin : 1.e4 est un préfixe → continuation attendue.
        let e4 = "e4".parse::<San>().unwrap().to_move(&pos).unwrap();
        let apres_e4 = pos.play(&e4).unwrap();
        assert!(
            coup_du_livre(&apres_e4, &mut rng).is_some(),
            "aucune continuation après 1.e4"
        );
    }

    /// Une position hors livre (après 1.h4) ne renvoie rien.
    #[test]
    fn coup_du_livre_hors_livre() {
        let mut rng = StdRng::seed_from_u64(7);
        let pos = Chess::default();
        let h4 = "h4".parse::<San>().unwrap().to_move(&pos).unwrap();
        let apres_h4 = pos.play(&h4).unwrap();
        assert!(coup_du_livre(&apres_h4, &mut rng).is_none());
    }

    /// tirage respecte grossièrement les probabilités demandées sur 2000
    /// tirages (p_ouverture = 0.5, p_finale = 0.25 → ~1000/500/500, tolérance
    /// large ±120), et les plis chauds suivent la famille (8/2/0).
    #[test]
    fn tirage_respecte_les_probabilites() {
        let mut rng = StdRng::seed_from_u64(2024);
        let (mut ouvertures, mut finales, mut initiales) = (0i32, 0i32, 0i32);
        for _ in 0..2000 {
            let d = tirage(&mut rng, 0.5, 0.25);
            if d.etiquette.starts_with("ouverture:") {
                assert_eq!(d.plis_chauds, 2);
                ouvertures += 1;
            } else if d.etiquette.starts_with("finale:") {
                assert_eq!(d.plis_chauds, 0);
                finales += 1;
            } else {
                assert_eq!(d.etiquette, "initiale");
                assert_eq!(d.plis_chauds, 8);
                assert_eq!(d.pos, Chess::default());
                initiales += 1;
            }
        }
        assert!((ouvertures - 1000).abs() <= 120, "ouvertures : {ouvertures}");
        assert!((finales - 500).abs() <= 120, "finales : {finales}");
        assert!((initiales - 500).abs() <= 120, "initiales : {initiales}");
    }
}
