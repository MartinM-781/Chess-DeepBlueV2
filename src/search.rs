//! Recherche « sérieuse » : négamax alpha-bêta à approfondissement itératif,
//! table de transposition, quiescence, tri des coups (coup TT, MVV-LVA,
//! killers, historique), élagage null-move. Les feuilles calmes sont évaluées
//! par le réseau (perspective du trait, [-1,1]) ; les mats sont exacts.
//!
//! RAFFINEMENTS (débrayables via `Recherche::mode_classique`, base des A/B) :
//! - LMR : les coups calmes tardifs sont sondés à profondeur réduite en
//!   fenêtre nulle, re-cherchés à pleine profondeur s'ils surprennent ;
//! - fenêtres d'aspiration à la racine (à partir de l'itération 3) ;
//! - SEE (échange statique sur la case d'arrivée, rayons X compris) : filtre
//!   les prises perdantes en quiescence — sauf celles qui donnent échec,
//!   repêchées pour que les mats restent visibles aux feuilles — et
//!   départage les prises au tri.
//!
//! ÉVALUATION INCRÉMENTALE (src/nnue.rs) : les feuilles ne repassent plus par
//! encode + forward complet — une pile d'accumulateurs de la couche 1 est
//! maintenue par deltas le long de la ligne explorée (pousse avant chaque
//! récursion, depousse au retour, pousse_null pour le null-move), et seule la
//! tête 512→64→1 est recalculée. Mêmes scores (à l'ordre des sommations f32
//! près), mêmes coups : la recherche est inchangée, seulement plus rapide.
//! Chemin de secours : champ `Recherche::utilise_nnue` (défaut true).
//!
//! LAZY SMP (champ `Recherche::threads`, défaut 1) : à N threads, N-1
//! assistants mènent le même approfondissement itératif sur la MÊME position
//! avec la table de transposition PARTAGÉE (sans verrous : deux mots
//! atomiques par case, validation cle ^ donnees — voir CaseTT), killers,
//! historique et piles d'accumulateurs PAR THREAD, profondeurs de départ et
//! fenêtres d'aspiration légèrement décalées. Le thread principal garde le
//! comportement mono-thread EXACT et rend le coup ; threads = 1 est bit à bit
//! le moteur historique (entraînement et gating inchangés).
//!
//! C'est l'étage 1 de la fusée « battre Deep Blue » : il sert à la fois à
//! JOUER (serveur, arène) et à FABRIQUER les étiquettes TD-leaf du self-play
//! (le score racine devient la cible d'apprentissage).
//!
//! Échelle des scores : réseau dans [-1, 1] ; mats à ±(SCORE_MAT - ply) pour
//! préférer les mats courts — SCORE_MAT domine largement l'échelle réseau.

use std::cmp::Reverse;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Bitboard, Board, Chess, Color, EnPassantMode, Move, Piece, Position, Role, Square};

use crate::features::N_FEATURES;
use crate::nn::Mlp;
use crate::nnue::{EvalIncrementale, PileAccus};
use crate::quant::{PileQuant, QuantNet};

pub const SCORE_MAT: f32 = 1000.0;

// L'interrupteur de secours de l'évaluation incrémentale (ex-constante
// USE_NNUE) est devenu le champ d'instance `Recherche::utilise_nnue`
// (défaut : true). Le passer à false rend le forward complet — comportement
// identique, seulement plus lent — utile pour isoler un bug de src/nnue.rs.

/// Au-delà de ce seuil (en valeur absolue) un score est un score de MAT :
/// les valeurs réseau vivent dans [-1, 1] et les mats dans
/// ±[SCORE_MAT - MAX_PLY - PROF_QUIESCENCE, SCORE_MAT], les deux échelles ne se
/// croisent jamais (1000 - 128 = 872 > 800).
const SEUIL_MAT: f32 = 800.0;

/// Profondeur maximale de l'approfondissement itératif (max_profondeur = 0 → ∞
/// borné par cette valeur, les limites de nœuds/temps arrêtant bien avant).
const PROF_MAX: u32 = 64;

/// Ply maximal de la recherche principale (garde-fou : au-delà on bascule en
/// quiescence). Doit rester > PROF_MAX et < 256 pour l'échelle des mats.
const MAX_PLY: usize = 120;

/// Profondeur maximale de la quiescence (prises en cascade bornées).
const PROF_QUIESCENCE: u32 = 8;

/// Réduction du null-move (profondeur fille = profondeur - 1 - R_NULL).
const R_NULL: u32 = 2;

/// Largeur de la fenêtre nulle du null-move : les scores sont des f32 continus,
/// on teste « >= beta » avec une fenêtre [beta - EPS_NUL, beta].
const EPS_NUL: f32 = 1e-3;

/// Demi-largeur initiale de la fenêtre d'aspiration à la racine (échelle
/// réseau [-1,1]). Doublée à chaque échec jusqu'à la fenêtre pleine.
const DELTA_ASPIRATION: f32 = 0.08;

/// Le chrono (Instant::now) n'est consulté que tous les ~1024 nœuds : un appel
/// d'horloge par nœud coûterait plus cher que le nœud lui-même.
const INTERVALLE_CHRONO: u64 = 1024;

// --- Table de transposition -------------------------------------------------

const DRAPEAU_VIDE: u8 = 0;
/// Score exact (la fenêtre n'a pas été coupée).
const DRAPEAU_EXACT: u8 = 1;
/// Borne inférieure : coupure bêta, le score réel est >= au score stocké.
const DRAPEAU_BORNE_INF: u8 = 2;
/// Borne supérieure : aucun coup n'a dépassé alpha, le score réel est <= stocké.
const DRAPEAU_BORNE_SUP: u8 = 3;

/// Coup compacté « aucun » : from=a1, to=a1, promotion=0 n'est jamais un coup
/// légal, 0 sert donc de sentinelle.
const COUP_AUCUN: u16 = 0;

/// Entrée de la table de transposition, forme DÉPAQUETÉE du mot de 64 bits
/// stocké dans une case (voir `paquette`).
///
/// ATTENTION AUX MATS (le piège classique) : un score de mat vaut
/// ±(SCORE_MAT - ply_racine), il dépend donc de la distance à la RACINE de la
/// recherche en cours. Stocké tel quel, il serait faux relu depuis un nœud à un
/// autre ply (le même mat semblerait plus proche ou plus lointain qu'il ne
/// l'est). On stocke donc les mats convertis en « distance au NŒUD » :
///   au stockage  : score_tt = score + ply (mats gagnants), score - ply (perdants)
///   à la relecture : score  = score_tt - ply', score_tt + ply'
/// si bien qu'une entrée écrite à ply 3 et relue à ply 7 rend un score de mat
/// correct vu de la nouvelle racine. Voir score_vers_tt / score_depuis_tt.
#[derive(Clone, Copy)]
struct EntreeTT {
    score: f32,
    coup: u16,
    profondeur: u8,
    drapeau: u8,
}

/// Compacte une entrée TT en 64 bits : score f32 (bits 0-31, via to_bits,
/// aller-retour EXACT) | coup (32-47) | profondeur (48-55) | drapeau (56-63).
/// 0 est la case vide (drapeau DRAPEAU_VIDE) : les tables s'allouent à zéro.
fn paquette(e: EntreeTT) -> u64 {
    u64::from(e.score.to_bits())
        | (u64::from(e.coup) << 32)
        | (u64::from(e.profondeur) << 48)
        | (u64::from(e.drapeau) << 56)
}

fn depaquette(d: u64) -> EntreeTT {
    EntreeTT {
        score: f32::from_bits(d as u32),
        coup: (d >> 32) as u16,
        profondeur: (d >> 48) as u8,
        drapeau: (d >> 56) as u8,
    }
}

/// Case de la table partagée : DEUX mots atomiques, `donnees` (l'entrée
/// compactée) et `cle_x = cle ^ donnees`. C'est le hachage « lockless »
/// standard des moteurs SMP : une course entre deux écritures peut entrelacer
/// deux entrées, mais la validation `cle_x ^ donnees == cle` de la lecture
/// échoue alors et l'entrée déchirée est simplement IGNORÉE — une course
/// donne une entrée invalide DÉTECTÉE, jamais une entrée corrompue acceptée.
/// Risque résiduel accepté : une fausse validation exigerait une coïncidence
/// XOR sur 64 bits (~2^-64 par course), du même ordre que les collisions
/// zobrist déjà tolérées par la table.
/// (Ceinture indépendante : un coup TT n'est de toute façon JAMAIS joué
/// directement — il est comparé à la liste LÉGALE de la position, via
/// `compacter` dans `cle_ordre`, et ne sert qu'au tri.)
struct CaseTT {
    cle_x: AtomicU64,
    donnees: AtomicU64,
}

/// 16 octets par case (deux u64, comme l'ancienne entrée nue) : une table de
/// 2^log2 cases pèse 2^(log2+4) octets — log2 26 → 1 Gio, 27 → 2 Gio,
/// 28 → 4 Gio, 29 → 8 Gio, 30 → 16 Gio (plafond).
const _: () = assert!(std::mem::size_of::<CaseTT>() == 16);

/// Taille mémoire en octets d'une table de 2^taille_log2 cases : pour les
/// gardes « TT plus grosse que la RAM » des harnais (match.exe).
pub fn octets_tt(taille_log2: u32) -> u64 {
    (std::mem::size_of::<CaseTT>() as u64) << taille_log2
}

/// Table de transposition, partageable entre threads (lazy SMP) : accès sans
/// verrous, cohérence par validation XOR (voir CaseTT). En mono-thread, la
/// sémantique (indexation, politique de remplacement, scores au bit près) est
/// STRICTEMENT celle de l'ancienne table à entrées nues.
struct TableTT {
    cases: Vec<CaseTT>,
    masque: u64,
}

impl TableTT {
    fn new(taille_log2: u32) -> Self {
        assert!(
            taille_log2 <= 30,
            "taille_tt_log2 déraisonnable (> 2^30 cases = 16 Gio)"
        );
        let n = 1usize << taille_log2;
        let mut cases = Vec::new();
        cases.resize_with(n, || CaseTT {
            cle_x: AtomicU64::new(0),
            donnees: AtomicU64::new(0),
        });
        TableTT { cases, masque: (n - 1) as u64 }
    }
}

/// Conversion d'un score « vu de la racine » en score « vu du nœud » pour le
/// stockage en TT (voir le commentaire d'EntreeTT : c'est LE piège des mats).
fn score_vers_tt(score: f32, ply: u32) -> f32 {
    if score > SEUIL_MAT {
        score + ply as f32
    } else if score < -SEUIL_MAT {
        score - ply as f32
    } else {
        score
    }
}

/// Conversion inverse à la relecture : le mat stocké « à distance du nœud »
/// redevient un score vu de la racine de la recherche EN COURS.
fn score_depuis_tt(score: f32, ply: u32) -> f32 {
    if score > SEUIL_MAT {
        score - ply as f32
    } else if score < -SEUIL_MAT {
        score + ply as f32
    } else {
        score
    }
}

// --- Aides génériques -------------------------------------------------------

/// Hachage zobrist 64 bits (même convention que selfplay/arena : mode Legal).
fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Compacte un coup en 16 bits : from (6) | to (6) | promotion (3).
/// Suffit à identifier un coup PARMI LES COUPS LÉGAUX d'une position (roque :
/// from = case du roi, to = case de la tour — paire unique elle aussi). Le coup
/// TT n'est jamais joué directement, seulement comparé à la liste légale.
fn compacter(m: &Move) -> u16 {
    let from = m.from().map_or(0, usize::from) as u16;
    let to = usize::from(m.to()) as u16;
    let promo = m.promotion().map_or(0u16, u16::from);
    from | (to << 6) | (promo << 12)
}

/// Valeur d'ordre d'un rôle : P=1, N=2, B=3, R=4, Q=5, K=6 (l'ordre suffit,
/// on ne s'en sert que pour trier).
fn valeur_role(r: shakmaty::Role) -> i32 {
    usize::from(r) as i32
}

/// Clé MVV-LVA d'un coup tactique : victime la plus grosse d'abord (facteur
/// 16 pour que la victime domine), agresseur le plus léger ensuite — c'est le
/// « valeur victime - valeur agresseur » du cahier des charges, à l'échelle
/// près qui évite qu'un gros agresseur déclasse une grosse victime.
/// Les promotions comptent comme tactiques (dame d'abord).
fn cle_mvv_lva(m: &Move) -> i32 {
    let mut cle = 0;
    if let Some(victime) = m.capture() {
        cle += 100_000 + 16 * valeur_role(victime) - valeur_role(m.role());
    }
    if let Some(promo) = m.promotion() {
        cle += 90_000 + 16 * valeur_role(promo);
    }
    cle
}

// --- SEE (Static Exchange Evaluation) ----------------------------------------

/// Valeur SEE d'un rôle, en centièmes de pion. Le roi vaut « très grand » :
/// il ne peut jamais être perdu avec profit, seulement conclure un échange
/// (la remontée minimax refuse d'elle-même toute « prise du roi »).
fn valeur_see(r: Role) -> i32 {
    match r {
        Role::Pawn => 100,
        Role::Knight => 300,
        Role::Bishop => 315,
        Role::Rook => 500,
        Role::Queen => 900,
        Role::King => 20_000,
    }
}

/// Moins cher attaquant de `camp` parmi `attaquants` (bitboard déjà filtré par
/// l'occupation courante) : rôles parcourus par valeur croissante.
fn moins_cher_attaquant(board: &Board, camp: Color, attaquants: Bitboard) -> Option<(Square, Role)> {
    if attaquants.is_empty() {
        return None;
    }
    for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King] {
        if let Some(sq) = (attaquants & board.by_piece(Piece { color: camp, role })).first() {
            return Some((sq, role));
        }
    }
    None
}

/// SEE : gain matériel espéré du coup `m` (centièmes de pion, point de vue du
/// camp qui joue), si les deux camps mènent l'échange optimal sur la case
/// d'arrivée.
///
/// MÉTHODE (« swap algorithm » itératif) :
/// 1. le coup est joué virtuellement : sa case de départ quitte l'occupation
///    (et la case du pion pris en passant, qui n'est PAS la case d'arrivée) ;
/// 2. tant que le camp au trait a des attaquants de la case, le MOINS CHER
///    recapture : gains[d] = valeur(occupant) - gains[d-1], le recapturant
///    devient l'occupant et sa case quitte l'occupation à son tour ;
/// 3. rayons X : `Board::attacks_to(case, camp, occ)` est recalculé à chaque
///    étage avec l'occupation RÉDUITE — tour/fou/dame alignés derrière une
///    pièce retirée réapparaissent d'eux-mêmes dans le bitboard (le résultat
///    est re-filtré par `& occ` pour écarter les pièces déjà consommées) ;
/// 4. remontée minimax : gains[d-1] = -max(-gains[d-1], gains[d]) — chaque
///    camp est libre de S'ARRÊTER de reprendre quand continuer le dessert.
///
/// Simplifications standard pour un SEE d'ordonnancement : clouages ignorés,
/// promotion comptée pour le coup INITIAL seulement (pas les re-captures),
/// roque → 0 (jamais une prise).
pub fn see(pos: &Chess, m: &Move) -> i32 {
    if m.is_castle() {
        return 0;
    }
    let Some(from) = m.from() else {
        return 0; // coups « posés » (variantes à réserve) : hors périmètre
    };
    let board = pos.board();
    let to = m.to();
    let mut occ = board.occupied().without(from);

    // Victime initiale ; le pion pris en passant est retiré À LA MAIN de
    // l'occupation (il ne se trouve pas sur la case d'arrivée).
    let victime = if m.is_en_passant() {
        occ = occ.without(Square::from_coords(to.file(), from.rank()));
        valeur_see(Role::Pawn)
    } else {
        m.capture().map_or(0, valeur_see)
    };

    // Pièce désormais posée sur la case : le rôle joué, ou la pièce promue
    // (la plus-value promo - pion s'ajoute au butin initial).
    let (mut occupant, bonus_promo) = match m.promotion() {
        Some(p) => (valeur_see(p), valeur_see(p) - valeur_see(Role::Pawn)),
        None => (valeur_see(m.role()), 0),
    };

    // 32 pièces au plus sur l'échiquier : 33 étages suffisent toujours.
    let mut gains = [0i32; 33];
    gains[0] = victime + bonus_promo;
    let mut d = 0usize;
    let mut camp = !pos.turn();
    while d + 1 < gains.len() {
        let attaquants = board.attacks_to(to, camp, occ) & occ;
        let Some((sq, role)) = moins_cher_attaquant(board, camp, attaquants) else {
            break;
        };
        d += 1;
        gains[d] = occupant - gains[d - 1];
        occupant = valeur_see(role);
        occ = occ.without(sq);
        camp = !camp;
    }
    while d > 0 {
        gains[d - 1] = -(-gains[d - 1]).max(gains[d]);
        d -= 1;
    }
    gains[0]
}

/// Le coup `m` donne-t-il échec ? Test EXACT (le coup est joué sur une
/// copie : échecs à la découverte compris). Appelé avec parcimonie — voir
/// quiesce : repêchage des seules prises see < 0 candidates au rejet.
fn donne_echec(pos: &Chess, m: &Move) -> bool {
    let mut fille = pos.clone();
    fille.play_unchecked(m);
    fille.is_check()
}

/// Le trait possède-t-il au moins une pièce qui ne soit ni pion ni roi ?
/// (Condition du null-move : évite les positions de zugzwang de finale de
/// pions, où « passer » est souvent la meilleure option réelle.)
fn a_piece_non_pion(pos: &Chess) -> bool {
    (pos.us() & !(pos.board().pawns() | pos.board().kings())).any()
}

fn idx_couleur(c: Color) -> usize {
    match c {
        Color::White => 0,
        Color::Black => 1,
    }
}

/// Limites d'un appel de recherche. 0 = pas de limite pour ce critère
/// (au moins un critère doit être non nul).
#[derive(Clone, Copy)]
pub struct Limites {
    pub max_noeuds: u64,
    pub max_profondeur: u32,
    pub movetime_ms: u64,
}

/// Paramètres de diversification d'un thread assistant du lazy SMP : chaque
/// assistant explore le même approfondissement itératif mais LÉGÈREMENT
/// décalé, pour peupler la TT partagée de sous-arbres différents.
#[derive(Clone, Copy)]
struct ParamsAssistant {
    /// Profondeur de la PREMIÈRE itération (le principal démarre toujours à 1).
    profondeur_depart: u32,
    /// Facteur (>= 1) sur la demi-largeur initiale des fenêtres d'aspiration.
    facteur_aspiration: f32,
}

/// Pose le drapeau d'arrêt partagé à sa destruction : les assistants sont
/// rappelés même si le thread principal sort par panique (sans quoi la
/// jointure implicite du scope les attendrait sans fin).
struct GardeArret<'a>(&'a AtomicBool);

impl Drop for GardeArret<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub struct Resultat {
    /// Meilleur coup (None seulement sans coup légal).
    pub coup: Option<Move>,
    /// Score du point de vue du trait ([-1,1] hors mats, ±(SCORE_MAT-ply) sinon).
    pub score: f32,
    /// Profondeur complète atteinte par l'approfondissement itératif.
    pub profondeur: u32,
    pub noeuds: u64,
    /// Scores racine (coup, score) de la DERNIÈRE itération complète —
    /// sert à l'échantillonnage en température du self-play.
    pub scores_racine: Vec<(Move, f32)>,
}

/// État persistant d'un chercheur : table de transposition, killers,
/// historique. UN chercheur par thread (rien de partagé). Possède son réseau
/// via Arc (partagé en lecture entre threads).
///
/// ATTENTION (rechargement du réseau) : `eval` COPIE les poids du réseau à la
/// construction. Si le réseau change (nouveau checkpoint chargé, Arc remplacé),
/// il FAUT reconstruire la `Recherche` — c'est déjà le cas partout dans le
/// projet : les bots (BotRecherche, self-play, duels de gating) sont recréés à
/// chaque partie ou à chaque cycle avec l'Arc du réseau courant.
pub struct Recherche {
    pub net: Arc<Mlp>,
    /// À `true`, DÉBRAYE les trois raffinements (LMR, fenêtres d'aspiration,
    /// SEE) : la recherche redevient strictement celle d'origine — même arbre,
    /// mêmes nœuds. C'est la base de comparaison des tests A/B. Défaut : false
    /// (raffinements actifs).
    pub mode_classique: bool,
    /// À `false`, DÉBRAYE l'évaluation incrémentale : chaque feuille repasse
    /// par encode + forward complet (comportement d'avant l'intégration NNUE,
    /// mêmes scores à l'ordre des sommations f32 près, seulement plus lent).
    /// Défaut : true. Sert d'échappatoire et de bras de comparaison au
    /// harnais forensique (src/bin/forensic.rs).
    pub utilise_nnue: bool,
    /// À `true`, active le chemin d'évaluation QUANTIZÉ (src/quant.rs :
    /// accumulateurs i32, têtes i8, AVX2) — PRIORITAIRE sur `utilise_nnue`.
    /// Défaut : false — drapeau off, comportement STRICTEMENT identique à
    /// avant ce chantier. L'erreur de quantization est bornée par la batterie
    /// de parité de quant.rs (max ≤ 0.05, moyenne ≤ 0.01 sur la sortie tanh).
    /// Réseau non quantizable (linéaire, poids hors domaine) : repli f32
    /// silencieux, jamais de panique.
    pub utilise_int8: bool,
    /// Nombre de threads de recherche (défaut 1). À 1, chemin STRICTEMENT
    /// identique au moteur historique — mêmes coups, mêmes nœuds : c'est le
    /// mode de l'entraînement, du self-play et du gating, qui ne changent
    /// pas. À N > 1, lazy SMP (voir cherche_smp) : réservé aux harnais de
    /// match/analyse.
    pub threads: u32,
    /// Poids réorganisés pour l'évaluation incrémentale. `None` = réseau sans
    /// couche cachée (les réseaux linéaires [773,1] de certains tests, où il
    /// n'y a de toute façon rien à accélérer) → forward complet d'office.
    eval: Option<EvalIncrementale>,
    /// Pile d'accumulateurs de la ligne en cours d'exploration, recréée à
    /// chaque appel de cherche() (racine posée sur la position de départ).
    /// Poussée/dépoussée par negamax et quiesce, en miroir de leur récursion.
    pile: Option<PileAccus>,
    /// Réseau quantizé (chemin int8), dérivé PARESSEUSEMENT du Mlp au premier
    /// cherche() avec `utilise_int8` actif — le drapeau off ne coûte rien.
    /// `None` tant que non construit OU si le réseau est hors domaine de
    /// quantization (voir QuantNet::depuis_mlp).
    quant: Option<QuantNet>,
    /// Vrai après la première tentative de dérivation (évite de retenter à
    /// chaque cherche() quand le réseau n'est pas quantizable).
    quant_tente: bool,
    /// Pile d'accumulateurs ENTIERS (chemin int8), pendante de `pile` —
    /// au plus une des deux est active pendant un cherche().
    pile_quant: Option<PileQuant>,
    /// Table de transposition : 2^taille_tt_log2 cases, index = cle & masque.
    /// Arc : les threads assistants du lazy SMP sondent et stockent la MÊME
    /// table (cohérence sans verrous, voir TableTT/CaseTT).
    tt: Arc<TableTT>,
    /// Deux killers par ply (coups calmes ayant produit une coupure bêta).
    killers: Vec<[u16; 2]>,
    /// Historique [couleur][from][to] aplati (2 × 64 × 64), incrémenté de
    /// profondeur² à chaque coupure bêta d'un coup calme.
    historique: Vec<u32>,
    /// Tampon d'encodage réutilisé pour chaque évaluation réseau.
    tampon: Vec<f32>,
    /// Clés zobrist de la LIGNE en cours d'exploration (racine → nœud) :
    /// détection des répétitions dans l'arbre. Pile poussée/poppée par negamax.
    chemin: Vec<u64>,
    // --- État d'un appel de cherche() (réinitialisé à chaque appel) ---
    noeuds: u64,
    stop: bool,
    /// Faux pendant l'itération 1 : elle est TOUJOURS menée à terme pour
    /// garantir un coup, quelles que soient les limites.
    limites_actives: bool,
    limite_noeuds: u64,
    fin: Option<Instant>,
    prochaine_verif_chrono: u64,
    /// Drapeau d'arrêt coopératif posé par le thread principal du lazy SMP
    /// (None hors SMP, donc toujours None sur un chercheur construit par
    /// `new`) : consulté par les assistants au même rythme que le chrono,
    /// tous les ~INTERVALLE_CHRONO nœuds.
    arret_partage: Option<Arc<AtomicBool>>,
    /// Statistiques TT du DERNIER cherche() — sondes et hits, tous threads
    /// confondus en SMP. Diagnostic des harnais ; aucun effet sur la
    /// recherche.
    pub tt_sondes: u64,
    pub tt_hits: u64,
}

impl Recherche {
    /// `taille_tt_log2` : nombre d'entrées de la table = 2^n (ex. 20 → ~1M
    /// d'entrées). La table est allouée une fois et réutilisée entre coups
    /// (les hits entre coups successifs sont une grosse part du gain).
    pub fn new(net: Arc<Mlp>, taille_tt_log2: u32) -> Self {
        Self::avec_table(net, Arc::new(TableTT::new(taille_tt_log2)))
    }

    /// Constructeur interne : chercheur posé sur une table EXISTANTE — le
    /// partage du lazy SMP (voir `assistant`). Tout le reste de l'état est
    /// neuf et PAR THREAD.
    fn avec_table(net: Arc<Mlp>, tt: Arc<TableTT>) -> Self {
        // Évaluation incrémentale construite UNE FOIS depuis le réseau (copie
        // des poids en colonnes) : voir le commentaire de la struct pour le
        // rechargement du réseau. Réseau sans couche cachée → forward complet.
        let eval = (net.sizes.len() >= 3).then(|| EvalIncrementale::new(&net));
        Recherche {
            net,
            mode_classique: false,
            utilise_nnue: true,
            utilise_int8: false,
            threads: 1,
            eval,
            pile: None,
            quant: None,
            quant_tente: false,
            pile_quant: None,
            tt,
            killers: vec![[COUP_AUCUN; 2]; MAX_PLY],
            historique: vec![0; 2 * 64 * 64],
            tampon: vec![0.0; N_FEATURES],
            chemin: Vec::with_capacity(MAX_PLY + 1),
            noeuds: 0,
            stop: false,
            limites_actives: false,
            limite_noeuds: u64::MAX,
            fin: None,
            prochaine_verif_chrono: 0,
            arret_partage: None,
            tt_sondes: 0,
            tt_hits: 0,
        }
    }

    /// Approfondissement itératif 1..=max jusqu'à épuisement des limites.
    /// Doit gérer : quiescence aux feuilles (prises + promotions, stand-pat
    /// réseau), null-move (R=2, pas en échec, matériel non-pion présent),
    /// tri coup TT > prises MVV-LVA > killers > historique, mats/pats exacts,
    /// nulles (50 coups, matériel insuffisant) à 0. La détection de répétition
    /// DANS l'arbre n'est pas exigée (les boucles de jeu l'arbitrent).
    /// `threads` = 1 (défaut) : recherche historique exacte ; > 1 : lazy SMP,
    /// même contrat, le coup rendu est celui du thread principal.
    pub fn cherche(&mut self, pos: &Chess, limites: Limites) -> Resultat {
        if self.threads <= 1 {
            return self
                .cherche_thread(pos, limites, None)
                .expect("l'itération 1 est toujours menée à terme");
        }
        self.cherche_smp(pos, limites)
    }

    /// Lazy SMP (threads >= 2) : N-1 assistants mènent le même
    /// approfondissement itératif sur la MÊME position, TT partagée, killers/
    /// historique/piles d'accumulateurs PAR THREAD, profondeurs de départ et
    /// fenêtres d'aspiration légèrement décalées (ParamsAssistant) pour
    /// diversifier les arbres. Le thread principal (self) garde exactement le
    /// comportement mono-thread et REND LE COUP ; à sa sortie, le drapeau
    /// d'arrêt partagé rappelle les assistants, qui le consultent au même
    /// rythme que le chrono. Les assistants sont des chercheurs NEUFS à
    /// chaque appel (quelques Mo de copies de poids — négligeable à la
    /// cadence d'un match ; leurs killers/historique repartent à zéro, la
    /// mémoire entre coups vit dans la TT partagée).
    fn cherche_smp(&mut self, pos: &Chess, limites: Limites) -> Resultat {
        let arret = Arc::new(AtomicBool::new(false));
        // Les assistants n'ont ni budget de nœuds ni profondeur propre : ils
        // s'arrêtent par le drapeau (et, en ceinture, par la même échéance de
        // chrono que le principal). PROF_MAX tient lieu de critère non nul
        // pour l'assertion de cherche_thread quand movetime_ms vaut 0
        // (appelant à nœuds ou profondeur fixes).
        let limites_assistant = Limites {
            max_noeuds: 0,
            max_profondeur: PROF_MAX,
            movetime_ms: limites.movetime_ms,
        };
        let mut assistants: Vec<Recherche> =
            (0..self.threads - 1).map(|_| self.assistant(&arret)).collect();
        let mut resultat = None;
        std::thread::scope(|s| {
            let garde = GardeArret(&arret);
            for (i, a) in assistants.iter_mut().enumerate() {
                // Diversification standard du lazy SMP : la moitié des
                // assistants commence un pli plus profond, et la demi-largeur
                // d'aspiration s'élargit par paliers (×1, ×1.5, ×2).
                let params = ParamsAssistant {
                    profondeur_depart: 1 + (i as u32 % 2),
                    facteur_aspiration: 1.0 + 0.5 * ((i / 2) % 3) as f32,
                };
                s.spawn(move || {
                    let _ = a.cherche_thread(pos, limites_assistant, Some(params));
                });
            }
            resultat = self.cherche_thread(pos, limites, None);
            // Le principal a fini : rappel des assistants, puis jointure
            // implicite en fin de scope. La garde couvre aussi la panique.
            drop(garde);
        });
        let mut resultat = resultat.expect("l'itération 1 est toujours menée à terme");
        // Comptes agrégés (diagnostic) : le coup, le score et scores_racine
        // restent ceux du thread principal.
        for a in &assistants {
            resultat.noeuds += a.noeuds;
            self.tt_sondes += a.tt_sondes;
            self.tt_hits += a.tt_hits;
        }
        resultat
    }

    /// Chercheur assistant du lazy SMP : état de recherche NEUF (killers,
    /// historique, piles), mêmes drapeaux d'évaluation, même réseau (Arc) et
    /// MÊME table de transposition (Arc) que `self`.
    fn assistant(&self, arret: &Arc<AtomicBool>) -> Recherche {
        let mut a = Recherche::avec_table(self.net.clone(), self.tt.clone());
        a.mode_classique = self.mode_classique;
        a.utilise_nnue = self.utilise_nnue;
        a.utilise_int8 = self.utilise_int8;
        a.arret_partage = Some(arret.clone());
        a
    }

    /// Corps de la recherche d'UN thread — l'ancien cherche(), paramétré.
    /// `assistant` : None pour le thread principal (comportement historique
    /// EXACT, l'itération 1 ignore les limites, résultat toujours Some) ;
    /// Some(params) pour un assistant SMP (limites actives dès la première
    /// itération, profondeur de départ et aspiration décalées, None si
    /// interrompu avant une première itération complète).
    fn cherche_thread(
        &mut self,
        pos: &Chess,
        limites: Limites,
        assistant: Option<ParamsAssistant>,
    ) -> Option<Resultat> {
        assert!(
            limites.max_noeuds > 0 || limites.max_profondeur > 0 || limites.movetime_ms > 0,
            "Limites : au moins un critère doit être non nul"
        );
        self.noeuds = 0;
        self.stop = false;
        self.limites_actives = false;
        self.limite_noeuds = if limites.max_noeuds == 0 { u64::MAX } else { limites.max_noeuds };
        self.fin = (limites.movetime_ms > 0)
            .then(|| Instant::now() + Duration::from_millis(limites.movetime_ms));
        self.prochaine_verif_chrono = INTERVALLE_CHRONO;
        self.tt_sondes = 0;
        self.tt_hits = 0;

        let coups = pos.legal_moves();
        if coups.is_empty() {
            // Position terminale : mat (le trait perd, ply 0) ou pat.
            return Some(Resultat {
                coup: None,
                score: if pos.is_check() { -SCORE_MAT } else { 0.0 },
                profondeur: 0,
                noeuds: 0,
                scores_racine: Vec::new(),
            });
        }
        // NB : si la position est déjà nulle « aux règles » (50 coups, matériel
        // insuffisant), l'arbitrage appartient aux boucles de jeu ; ici on rend
        // quand même un coup (contrat : None seulement sans coup légal).

        let prof_max = if limites.max_profondeur == 0 {
            PROF_MAX
        } else {
            limites.max_profondeur.min(PROF_MAX)
        };

        // Pile d'accumulateurs posée sur la position de départ : une par appel
        // (la racine change à chaque coup joué). Encodage complet des deux
        // perspectives ici, puis uniquement des deltas dans l'arbre.
        // Chemin int8 prioritaire quand `utilise_int8` est actif : le réseau
        // quantizé est dérivé du Mlp au PREMIER appel (une fois), puis la
        // pile entière remplace la pile f32. Réseau non quantizable →
        // `quant` reste None et on retombe sur les chemins f32 ci-dessous.
        if self.utilise_int8 && !self.quant_tente {
            self.quant_tente = true;
            self.quant = QuantNet::depuis_mlp(&self.net);
        }
        self.pile_quant = if self.utilise_int8 {
            self.quant.as_ref().map(|q| q.racine(pos))
        } else {
            None
        };
        self.pile = if self.utilise_nnue && self.pile_quant.is_none() {
            self.eval.as_ref().map(|e| e.racine(pos))
        } else {
            // Incrémental débrayé (ou chemin int8 actif) : pile f32 absente →
            // evaluer() et les pile_pousse/depousse retombent d'eux-mêmes sur
            // l'autre chemin (no-ops côté pile f32).
            None
        };

        // Ordre racine initial : coup TT de la recherche précédente (grosse
        // source de gain entre coups successifs d'une même partie), puis
        // tactiques MVV-LVA, puis historique.
        let cle_racine = zobrist(pos);
        // La racine ouvre la ligne courante (les nœuds la prolongent) ; un
        // arrêt en plein arbre peut laisser des résidus → on repart propre.
        self.chemin.clear();
        self.chemin.push(cle_racine);
        let coup_tt_racine = self.sonde(cle_racine).map_or(COUP_AUCUN, |e| e.coup);
        let couleur = pos.turn();
        let mut ordre: Vec<Move> = coups.iter().cloned().collect();
        ordre.sort_by_cached_key(|m| Reverse(self.cle_ordre(pos, m, coup_tt_racine, 0, couleur)));

        // Dernière itération COMPLÈTE : (meilleur coup, score, profondeur, scores racine).
        let mut complete: Option<(Move, f32, u32, Vec<(Move, f32)>)> = None;

        // Assistant SMP : première itération décalée (diversification) et
        // limites actives d'emblée — il n'a pas à garantir un coup, le
        // principal s'en charge. Principal : départ à 1, itération 1 hors
        // limites, comme toujours.
        let depart = assistant.map_or(1, |p| p.profondeur_depart.clamp(1, prof_max));
        'iterations: for d in depart..=prof_max {
            // L'itération 1 (du principal) ignore les limites : il faut
            // TOUJOURS un coup.
            self.limites_actives = assistant.is_some() || d > 1;
            if self.limites_actives && self.budget_epuise() {
                break;
            }

            // Fenêtre d'aspiration à la racine (raffinement débrayé en mode
            // classique) : à partir de l'itération 3, on parie que le score
            // restera proche de celui de l'itération précédente — fenêtre
            // [prec - DELTA_ASPIRATION, prec + DELTA_ASPIRATION] — et en cas
            // d'échec (fail-low/high) la passe est REFAITE en doublant la
            // demi-largeur du côté fautif jusqu'à la fenêtre pleine. Scores de
            // MAT (|prec| > SEUIL_MAT) : fenêtre pleine directement, ±0.08
            // n'a aucun sens à l'échelle des mats.
            let score_prec = complete.as_ref().map(|c| c.1);
            let aspiration = !self.mode_classique
                && d >= 3
                && score_prec.is_some_and(|s| s.abs() <= SEUIL_MAT);
            // Assistant SMP : fenêtre initiale élargie d'un facteur propre au
            // thread (diversification) ; principal : DELTA_ASPIRATION tel quel.
            let mut demi = match assistant {
                Some(p) => DELTA_ASPIRATION * p.facteur_aspiration,
                None => DELTA_ASPIRATION,
            };
            let (mut fen_bas, mut fen_haut) = if aspiration {
                let s = score_prec.expect("aspiration exige une itération précédente");
                (s - demi, s + demi)
            } else {
                (f32::NEG_INFINITY, f32::INFINITY)
            };

            // Une « passe » racine par fenêtre ; en fenêtre pleine (mode
            // classique, d < 3, mats) la boucle ne fait qu'un seul tour et le
            // corps est strictement la boucle racine d'origine.
            let (best, meilleur, scores_iter) = loop {
                let mut alpha = fen_bas;
                let mut best = f32::NEG_INFINITY;
                let mut meilleur: Option<Move> = None;
                let mut scores_iter: Vec<(Move, f32)> = Vec::with_capacity(ordre.len());
                let mut fail_high = false;

                for m in &ordre {
                    let mut fille = pos.clone();
                    fille.play_unchecked(m);
                    // Fenêtre racine : alpha monte, bêta est le plafond
                    // d'aspiration (infini hors aspiration : tous les coups
                    // racine sont cherchés). Le meilleur score est exact ;
                    // les autres peuvent n'être que des bornes supérieures
                    // (fail-soft) — suffisant pour l'échantillonnage en
                    // température, qui ne sert qu'à l'exploration.
                    self.pile_pousse(pos, m);
                    let v = -self.negamax(&fille, d - 1, 1, -fen_haut, -alpha, false);
                    self.pile_depousse();
                    if self.stop {
                        break 'iterations; // itération jetée : on garde la dernière complète
                    }
                    // Chrono aussi consulté entre deux coups racine : la granularité
                    // en nœuds (INTERVALLE_CHRONO) peut être trop grossière quand le
                    // réseau est lent ; ici l'appel d'horloge est gratuit à l'échelle
                    // d'un sous-arbre. Sans incidence sur le déterminisme à budget de
                    // nœuds fixe (fin = None dans ce cas). Même consultation du
                    // drapeau d'arrêt partagé pour un assistant SMP (None sinon).
                    if self.limites_actives
                        && (self.fin.is_some_and(|f| Instant::now() >= f)
                            || self.arret_partage.as_ref().is_some_and(|a| a.load(Ordering::Relaxed)))
                    {
                        self.stop = true;
                        break 'iterations;
                    }
                    scores_iter.push((m.clone(), v));
                    if v > best {
                        best = v;
                        meilleur = Some(m.clone());
                    }
                    if v > alpha {
                        alpha = v;
                    }
                    if v >= fen_haut {
                        // Fail-high d'aspiration : le score crève le plafond,
                        // la passe est invalide — élargir sans finir le tour.
                        fail_high = true;
                        break;
                    }
                }

                if fail_high || best <= fen_bas {
                    // Échec d'aspiration (impossible en fenêtre pleine : les
                    // scores sont finis et fen_haut infini). Doublement du
                    // côté fautif ; une borne déjà en zone de MAT saute
                    // directement en fenêtre pleine.
                    let s = score_prec.expect("échec d'aspiration sans fenêtre étroite");
                    demi *= 2.0;
                    let pleine = demi >= 2.0 || best.abs() > SEUIL_MAT;
                    if fail_high {
                        // Le coup fautif mène la re-passe (il y échouera haut
                        // ou s'y prouvera le meilleur au plus vite).
                        if let Some(m) = &meilleur {
                            if let Some(i) = ordre.iter().position(|o| o == m) {
                                ordre[..=i].rotate_right(1);
                            }
                        }
                        fen_haut = if pleine { f32::INFINITY } else { s + demi };
                    } else {
                        fen_bas = if pleine { f32::NEG_INFINITY } else { s - demi };
                    }
                    continue;
                }
                break (best, meilleur, scores_iter);
            };

            // Réordonne la racine pour l'itération suivante : meilleurs scores
            // d'abord (tri stable : le meilleur reste devant ses ex æquo).
            let mut tri = scores_iter.clone();
            tri.sort_by(|a, b| b.1.total_cmp(&a.1));
            ordre = tri.into_iter().map(|(m, _)| m).collect();

            let coup = meilleur.expect("itération complète avec coups légaux");
            self.stocke(cle_racine, d, DRAPEAU_EXACT, score_vers_tt(best, 0), compacter(&coup));
            let mat_trouve = best.abs() > SEUIL_MAT;
            complete = Some((coup, best, d, scores_iter));
            if mat_trouve {
                break; // mat prouvé dans l'horizon : inutile de creuser
            }
        }

        // Principal : Some garanti (son itération 1 ignore les limites) —
        // l'expect vit chez les appelants. Assistant : None si le drapeau
        // d'arrêt l'a interrompu avant une première itération complète.
        let (coup, score, profondeur, scores_racine) = complete?;
        Some(Resultat {
            coup: Some(coup),
            score,
            profondeur,
            noeuds: self.noeuds,
            scores_racine,
        })
    }

    /// À appeler entre deux PARTIES (pas entre deux coups) : vide TT,
    /// killers et historique. (La table étant partagée en SMP, la remise à
    /// zéro vaut pour tous les threads — les assistants d'un cherche() sont
    /// de toute façon éphémères.)
    pub fn nouvelle_partie(&mut self) {
        for case in &self.tt.cases {
            case.cle_x.store(0, Ordering::Relaxed);
            case.donnees.store(0, Ordering::Relaxed);
        }
        self.killers.fill([COUP_AUCUN; 2]);
        self.historique.fill(0);
    }

    // --- Négamax alpha-bêta (fail-soft) --------------------------------------

    fn negamax(
        &mut self,
        pos: &Chess,
        profondeur: u32,
        ply: u32,
        mut alpha: f32,
        beta: f32,
        null_interdit: bool,
    ) -> f32 {
        // Feuille : quiescence (elle fait son propre comptage et ses propres
        // verdicts mat/pat/nulle, avec sa propre génération de coups).
        if profondeur == 0 || ply as usize >= MAX_PLY {
            return self.quiesce(pos, ply, PROF_QUIESCENCE, alpha, beta);
        }

        self.noeuds += 1;
        if self.verifier_arret() {
            return 0.0; // valeur jetée : l'itération interrompue est abandonnée
        }

        // Génération COMPLÈTE : tous les coups sont joués ici, le verdict de
        // mat/pat est donc exact. Mat/pat testés AVANT la règle des 50 coups,
        // comme partout dans le projet : un mat délivré pile au 100e
        // demi-coup reste un mat.
        let coups = pos.legal_moves();
        let en_echec = pos.is_check();
        if coups.is_empty() {
            return if en_echec { -(SCORE_MAT - ply as f32) } else { 0.0 };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
            return 0.0;
        }

        let cle = zobrist(pos);

        // Répétition DANS LA LIGNE explorée (la position est déjà apparue
        // entre la racine et ce nœud) → nulle, 0.0. Le contrat n'exige pas la
        // détection de la 3e répétition RÉELLE (les boucles de jeu
        // l'arbitrent), mais sans CE test un camp gagnant « mélange » ses
        // pièces vers la répétition au lieu de progresser — la 2e occurrence
        // dans une même ligne vaut nulle, comme dans tous les moteurs. Ce
        // score dépend du chemin : il est rendu AVANT la sonde TT et n'est
        // jamais stocké (le reste de contamination indirecte via les
        // sous-arbres est le compromis GHI standard).
        if self.chemin.contains(&cle) {
            return 0.0;
        }

        // Sonde de la table de transposition. NB : la clé zobrist ignore le
        // compteur des 50 coups — une entrée peut donc court-circuiter un
        // sous-arbre qui aurait buté sur la règle ; c'est le compromis
        // standard de tous les moteurs, accepté ici aussi.
        let mut coup_tt = COUP_AUCUN;
        if let Some(e) = self.sonde(cle) {
            coup_tt = e.coup;
            if u32::from(e.profondeur) >= profondeur {
                // Relecture avec ré-ajustement du ply : voir EntreeTT (mats).
                let s = score_depuis_tt(e.score, ply);
                match e.drapeau {
                    DRAPEAU_EXACT => return s,
                    DRAPEAU_BORNE_INF if s >= beta => return s,
                    DRAPEAU_BORNE_SUP if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let alpha_orig = alpha;
        // La clé du nœud rejoint la ligne courante le temps d'explorer ses
        // enfants (null-move compris) ; UNIQUE pop après le bloc, quel que
        // soit le chemin de sortie.
        self.chemin.push(cle);
        let (best, meilleur_coup, stocker) = 'corps: {
            // Null-move (R=2) : si « passer » suffit déjà à couper, la
            // position est si bonne qu'on s'épargne la recherche complète.
            // Conditions : pas en échec, profondeur >= 3, du matériel non-pion
            // (zugzwang), jamais deux nulls consécutifs, une bêta finie
            // (sinon rien à couper) et une bêta HORS zone de mat (garde
            // standard des moteurs : si beta > SEUIL_MAT — fenêtre d'un
            // sous-arbre de preuve de mat — le rabattement anti-mat du
            // fail-high renverrait beta, c'est-à-dire un score de MAT non
            // prouvé, et une défense réelle pourrait être élaguée en cas de
            // zugzwang).
            if !null_interdit
                && profondeur >= 1 + R_NULL
                && !en_echec
                && beta.is_finite()
                && beta < SEUIL_MAT
                && a_piece_non_pion(pos)
            {
                if let Ok(passe) = pos.clone().swap_turn() {
                    // Null-move sur la pile : position inchangée, seules les
                    // perspectives s'échangeront à l'évaluation (sommet dupliqué).
                    self.pile_pousse_null();
                    let v = -self.negamax(
                        &passe,
                        profondeur - 1 - R_NULL,
                        ply + 1,
                        -beta,
                        -beta + EPS_NUL,
                        true,
                    );
                    self.pile_depousse();
                    if self.stop {
                        break 'corps (0.0, COUP_AUCUN, false);
                    }
                    if v >= beta {
                        // Jamais de score de MAT non prouvé issu d'un
                        // null-move : on rabat sur beta (mater un adversaire
                        // qui passe son tour ne prouve pas un mat réel) —
                        // et beta < SEUIL_MAT (garde d'entrée), donc la
                        // valeur rendue est toujours hors zone de mat.
                        break 'corps (if v > SEUIL_MAT { beta } else { v }, COUP_AUCUN, false);
                    }
                }
            }

            // Tri : coup TT > prises (SEE en mode raffiné, MVV-LVA en mode
            // classique) > killers du ply > historique.
            let mut ordonnes: Vec<(i32, &Move)> = coups
                .iter()
                .map(|m| (self.cle_ordre(pos, m, coup_tt, ply, pos.turn()), m))
                .collect();
            ordonnes.sort_unstable_by_key(|(k, _)| Reverse(*k));

            let mut best = f32::NEG_INFINITY;
            let mut meilleur_coup = COUP_AUCUN;
            let mut examines = 0u32; // coups déjà cherchés dans CE nœud

            for (_, m) in &ordonnes {
                let mut fille = pos.clone();
                fille.play_unchecked(m);
                // LMR (Late Move Reductions, débrayé en mode classique) : un
                // coup CALME tardif — 4e examiné ou au-delà, profondeur >= 3,
                // ni prise ni promotion, pas en échec avant le coup et n'en
                // donnant pas, jamais le coup TT ni un killer — est d'abord
                // SONDÉ à profondeur réduite (1, puis 2 à partir du 8e coup)
                // en fenêtre nulle : le tri étant bon, il échoue presque
                // toujours sous alpha pour une fraction du prix. Si le sondage
                // dépasse alpha, RE-recherche à pleine profondeur et pleine
                // fenêtre : aucune décision n'est prise sur la seule foi d'une
                // recherche réduite. (alpha est toujours fini dès le 2e coup
                // d'un nœud ; la garde is_finite est une ceinture.)
                let reduction = if !self.mode_classique
                    && examines >= 3
                    && profondeur >= 3
                    && !en_echec
                    && !m.is_capture()
                    && !m.is_promotion()
                    && alpha.is_finite()
                {
                    let c = compacter(m);
                    let k = &self.killers[ply as usize];
                    if c != coup_tt && c != k[0] && c != k[1] && !fille.is_check() {
                        if examines >= 7 { 2 } else { 1 }
                    } else {
                        0
                    }
                } else {
                    0
                };
                // Deltas du coup sur la pile (2 à 4 features, 2 perspectives),
                // dépilés au retour quel que soit le chemin de sortie.
                self.pile_pousse(pos, m);
                let mut v = if reduction > 0 {
                    // Sondage réduit, fenêtre nulle [alpha, alpha + EPS_NUL].
                    -self.negamax(
                        &fille,
                        profondeur - 1 - reduction,
                        ply + 1,
                        -alpha - EPS_NUL,
                        -alpha,
                        false,
                    )
                } else {
                    -self.negamax(&fille, profondeur - 1, ply + 1, -beta, -alpha, false)
                };
                if reduction > 0 && !self.stop && v > alpha {
                    // Le coup réduit promet : re-recherche complète.
                    v = -self.negamax(&fille, profondeur - 1, ply + 1, -beta, -alpha, false);
                }
                self.pile_depousse();
                if self.stop {
                    // Valeur jetée, surtout ne rien stocker en TT.
                    break 'corps (best, COUP_AUCUN, false);
                }
                examines += 1;
                if v > best {
                    best = v;
                    meilleur_coup = compacter(m);
                    if v > alpha {
                        alpha = v;
                        if alpha >= beta {
                            // Coupure bêta : killers + historique (coups calmes).
                            if !m.is_capture() && !m.is_promotion() {
                                self.note_killer(ply, meilleur_coup);
                                self.note_historique(pos.turn(), m, profondeur);
                            }
                            break;
                        }
                    }
                }
            }
            (best, meilleur_coup, true)
        };
        self.chemin.pop();
        if !stocker {
            return best;
        }

        // Stockage TT : drapeau selon la fenêtre d'ORIGINE, score de mat
        // converti en distance au nœud (score_vers_tt, voir EntreeTT).
        let drapeau = if best >= beta {
            DRAPEAU_BORNE_INF
        } else if best <= alpha_orig {
            DRAPEAU_BORNE_SUP
        } else {
            DRAPEAU_EXACT
        };
        self.stocke(cle, profondeur, drapeau, score_vers_tt(best, ply), meilleur_coup);
        best
    }

    // --- Quiescence ----------------------------------------------------------

    /// Hors échec : stand-pat réseau puis prises et promotions uniquement,
    /// triées MVV-LVA, alpha-bêta fail-soft. EN échec : pas de stand-pat (on
    /// ne peut pas « passer » sous échec), on cherche TOUTES les évasions — la
    /// liste légale complète EST la liste des évasions, donc le mat est déclaré
    /// EXACTEMENT même ici (le piège « conclure au mat à court de prises »
    /// ne peut pas se produire : à court de prises hors échec on rend le
    /// stand-pat, et en échec tous les coups sont essayés).
    fn quiesce(&mut self, pos: &Chess, ply: u32, prof_restante: u32, mut alpha: f32, beta: f32) -> f32 {
        self.noeuds += 1;
        if self.verifier_arret() {
            return 0.0;
        }

        // Liste légale COMPLÈTE (c'est ainsi que shakmaty produit les prises) :
        // vide → mat ou pat, verdict exact même en quiescence.
        let coups = pos.legal_moves();
        let en_echec = pos.is_check();
        if coups.is_empty() {
            return if en_echec { -(SCORE_MAT - ply as f32) } else { 0.0 };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
            return 0.0;
        }
        let mut best;
        if en_echec {
            // Garde-fou de profondeur : une cascade d'échecs ne peut pas
            // s'étendre indéfiniment, on retombe sur l'évaluation brute.
            if prof_restante == 0 {
                return self.evaluer(pos);
            }
            best = f32::NEG_INFINITY; // pas de stand-pat sous échec
        } else {
            // Stand-pat : « je peux m'abstenir de prendre » — l'évaluation
            // réseau de la position telle quelle, perspective du trait.
            let stand_pat = self.evaluer(pos);
            if prof_restante == 0 {
                return stand_pat; // profondeur de quiescence épuisée
            }
            if stand_pat >= beta {
                return stand_pat; // fail-soft
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            best = stand_pat;
        }

        // En échec : toutes les évasions ; sinon tactiques seulement — et, en
        // mode raffiné, les prises que le SEE juge PERDANTES (see < 0) sont
        // ignorées d'office : la quiescence existe pour solder les échanges,
        // pas pour explorer les sacrifices (les promotions calmes restent
        // examinées ; sous échec, AUCUN filtre : toutes les évasions comptent
        // pour garder les verdicts de mat exacts).
        // REPÊCHAGE (audit « le mat est vu ») : une prise see < 0 qui DONNE
        // ÉCHEC échappe au filtre. Le SEE est aveugle aux suites forcées :
        // sur une prise qui MATE une case « défendue » (ex. Qxh7# du test
        // 7 bis), le défenseur qu'il compte ne reprendra jamais — la partie
        // est finie — et le filtre cachait ce mat aux feuilles, que le mode
        // classique voyait. Le test donne_echec (play + is_check) n'est payé
        // que par les prises déjà condamnées (see < 0, court-circuit du &&),
        // une petite minorité.
        // Tri MVV-LVA, plus grosse victime d'abord (les évasions calmes,
        // clé 0, passent après les prises).
        let filtre_see = !self.mode_classique && !en_echec;
        let mut a_jouer: Vec<(i32, &Move)> = coups
            .iter()
            .filter(|m| en_echec || m.is_capture() || m.is_promotion())
            .filter(|m| {
                !(filtre_see && m.is_capture() && see(pos, m) < 0 && !donne_echec(pos, m))
            })
            .map(|m| (cle_mvv_lva(m), m))
            .collect();
        a_jouer.sort_unstable_by_key(|(k, _)| Reverse(*k));

        for (_, m) in &a_jouer {
            let mut fille = pos.clone();
            fille.play_unchecked(m);
            self.pile_pousse(pos, m);
            let v = -self.quiesce(&fille, ply + 1, prof_restante - 1, -beta, -alpha);
            self.pile_depousse();
            if self.stop {
                return best; // valeur jetée de toute façon (itération abandonnée)
            }
            if v > best {
                best = v;
                if v > alpha {
                    alpha = v;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    // --- Évaluation réseau ---------------------------------------------------

    /// Évaluation d'une feuille, perspective du trait, sortie dans [-1, 1].
    ///
    /// Chemin normal : lecture du sommet de la pile d'accumulateurs (`pos`
    /// DOIT être la position du sommet — garanti par la discipline
    /// pousse/depousse de negamax et quiesce) + tête 512→64→1.
    /// Chemin de secours (eval None, ou pile None quand utilise_nnue est à
    /// false) : encode + forward complet, identique au comportement d'avant
    /// l'intégration NNUE.
    fn evaluer(&mut self, pos: &Chess) -> f32 {
        // Chemin int8 (prioritaire, actif seulement si cherche() a posé la
        // pile quantizée) : lecture du sommet entier + têtes i8.
        if let (Some(quant), Some(pile_q)) = (self.quant.as_ref(), self.pile_quant.as_ref()) {
            let v = pile_q.evalue(quant, pos);
            // Garde débug échantillonnée (même cadence que le chemin f32) :
            // l'écart LÉGITIME de quantization reste ≤ ~0.05 (batterie de
            // parité de quant.rs) ; un bug d'indexation ou de pile décale
            // d'un ordre de grandeur. Tolérance 0.15 = 3× le seuil de la
            // batterie : ne se déclenche jamais sur le bruit de quantization.
            #[cfg(debug_assertions)]
            if self.noeuds % 4096 == 1 {
                let reference = crate::nn::evalue_position(&self.net, pos, &mut self.tampon);
                debug_assert!(
                    (v - reference).abs() <= 0.15,
                    "divergence int8 au nœud {} : quantizé {v} vs forward complet {reference}",
                    self.noeuds
                );
            }
            return v;
        }
        if let (Some(eval), Some(pile)) = (self.eval.as_ref(), self.pile.as_ref()) {
            let v = pile.evalue(eval, pos);
            // Parité échantillonnée, MODE DEBUG UNIQUEMENT : 1 nœud sur 4096
            // (dont le tout premier de chaque recherche) est recalculé par le
            // forward complet. Si l'assertion se déclenche, la pile a divergé
            // (delta de coup faux, pousse/depousse déséquilibrés...) :
            // basculer utilise_nnue à false (champ de Recherche) pour revenir
            // au forward complet le temps de corriger src/nnue.rs. Tolérance
            // 1e-3 : la dérive légitime (ordre des sommations f32 accumulées
            // le long d'une partie) reste ~1e-5, un vrai bug de delta décale
            // les pré-activations de plusieurs ordres de grandeur au-dessus.
            #[cfg(debug_assertions)]
            if self.noeuds % 4096 == 1 {
                // Référence par le répartiteur de schéma : valide pour les
                // réseaux Classique773 COMME RoiZones8 (encode+forward_one en
                // dur paniquerait avec un réseau 6149).
                let reference = crate::nn::evalue_position(&self.net, pos, &mut self.tampon);
                debug_assert!(
                    (v - reference).abs() <= 1e-3,
                    "divergence NNUE au nœud {} : incrémental {v} vs forward complet {reference}",
                    self.noeuds
                );
            }
            return v;
        }
        // Répartiteur de schéma (et non encode + forward_one en dur, qui ne
        // vaut que pour Classique773 et paniquerait avec un réseau RoiZones8
        // 6149) : bit-à-bit identique au chemin historique pour Classique773.
        crate::nn::evalue_position(&self.net, pos, &mut self.tampon)
    }

    // --- Pile d'accumulateurs (no-ops en chemin de secours) ------------------

    /// Empile les deltas du coup `m` joué depuis `pos_avant` (les deux
    /// perspectives). À appeler juste avant CHAQUE récursion sur une fille.
    #[inline]
    fn pile_pousse(&mut self, pos_avant: &Chess, m: &Move) {
        if let (Some(pile), Some(eval)) = (self.pile.as_mut(), self.eval.as_ref()) {
            pile.pousse(eval, pos_avant, m);
        }
        if let (Some(pile_q), Some(quant)) = (self.pile_quant.as_mut(), self.quant.as_ref()) {
            pile_q.pousse(quant, pos_avant, m);
        }
    }

    /// Empile un null-move (sommet dupliqué, perspectives échangées à la
    /// lecture).
    #[inline]
    fn pile_pousse_null(&mut self) {
        if let Some(pile) = self.pile.as_mut() {
            pile.pousse_null();
        }
        if let Some(pile_q) = self.pile_quant.as_mut() {
            pile_q.pousse_null();
        }
    }

    /// Dépile un étage. À appeler au retour de CHAQUE récursion, avant tout
    /// branchement (stop, coupure...) pour rester en miroir exact des pousse.
    #[inline]
    fn pile_depousse(&mut self) {
        if let Some(pile) = self.pile.as_mut() {
            pile.depousse();
        }
        if let Some(pile_q) = self.pile_quant.as_mut() {
            pile_q.depousse();
        }
    }

    /// (Tests uniquement) Force le chemin de secours « forward complet » :
    /// sert de référence aux tests de parité NNUE et au AVANT des benchs.
    #[cfg(test)]
    fn force_forward_complet(&mut self) {
        self.eval = None;
        self.pile = None;
        self.utilise_int8 = false;
        self.quant = None;
        self.pile_quant = None;
    }

    // --- Limites -------------------------------------------------------------

    /// Vrai si la recherche doit s'arrêter. Le budget de nœuds est testé à
    /// CHAQUE nœud (comparaison d'entiers, gratuite et déterministe) ; le
    /// chrono — et, pour un assistant SMP, le drapeau d'arrêt partagé — tous
    /// les ~INTERVALLE_CHRONO nœuds seulement.
    fn verifier_arret(&mut self) -> bool {
        if self.stop {
            return true;
        }
        if !self.limites_actives {
            return false; // itération 1 : toujours menée à terme
        }
        if self.noeuds >= self.limite_noeuds {
            self.stop = true;
            return true;
        }
        if (self.fin.is_some() || self.arret_partage.is_some())
            && self.noeuds >= self.prochaine_verif_chrono
        {
            self.prochaine_verif_chrono = self.noeuds + INTERVALLE_CHRONO;
            if self.fin.is_some_and(|f| Instant::now() >= f) {
                self.stop = true;
                return true;
            }
            if self.arret_partage.as_ref().is_some_and(|a| a.load(Ordering::Relaxed)) {
                self.stop = true;
                return true;
            }
        }
        false
    }

    /// Test direct (entre deux itérations) : budget déjà consommé ? (Pour un
    /// assistant SMP, le drapeau d'arrêt partagé compte comme un budget.)
    fn budget_epuise(&self) -> bool {
        self.noeuds >= self.limite_noeuds
            || self.fin.is_some_and(|f| Instant::now() >= f)
            || self.arret_partage.as_ref().is_some_and(|a| a.load(Ordering::Relaxed))
    }

    // --- Table de transposition ----------------------------------------------

    fn sonde(&mut self, cle: u64) -> Option<EntreeTT> {
        let case = &self.tt.cases[(cle & self.tt.masque) as usize];
        let cle_x = case.cle_x.load(Ordering::Relaxed);
        let donnees = case.donnees.load(Ordering::Relaxed);
        self.tt_sondes += 1;
        let e = depaquette(donnees);
        // Validation lockless : drapeau non vide ET clé reconstituée exacte
        // (cle_x ^ donnees). Une écriture déchirée par une course entre
        // threads échoue ici et vaut simplement « absent ».
        if e.drapeau != DRAPEAU_VIDE && cle_x ^ donnees == cle {
            self.tt_hits += 1;
            Some(e)
        } else {
            None
        }
    }

    /// Remplacement : case vide, clé différente, ou profondeur >= existante
    /// (une recherche plus profonde de la même position est plus fiable).
    /// Lecture-décision-écriture NON atomique : entre threads, le pire cas
    /// est un remplacement « injuste » ou perdu — jamais une entrée corrompue
    /// acceptée (la validation XOR de sonde() détecte tout entrelacement).
    fn stocke(&mut self, cle: u64, profondeur: u32, drapeau: u8, score: f32, coup: u16) {
        let case = &self.tt.cases[(cle & self.tt.masque) as usize];
        let cle_x = case.cle_x.load(Ordering::Relaxed);
        let ancien = case.donnees.load(Ordering::Relaxed);
        let e = depaquette(ancien);
        if e.drapeau == DRAPEAU_VIDE
            || cle_x ^ ancien != cle
            || profondeur >= u32::from(e.profondeur)
        {
            let d = paquette(EntreeTT {
                score,
                coup,
                profondeur: profondeur.min(255) as u8,
                drapeau,
            });
            case.donnees.store(d, Ordering::Relaxed);
            case.cle_x.store(cle ^ d, Ordering::Relaxed);
        }
    }

    // --- Tri des coups -------------------------------------------------------

    /// Clé de tri décroissante d'un coup dans la recherche principale.
    /// Mode classique : coup TT (1 000 000) > tactiques MVV-LVA
    /// (~100 000-190 000) > promotions calmes (~90 000) > killers
    /// (80 000 / 79 000) > historique (0..=60 000).
    /// Mode raffiné : les PRISES sont départagées par le SEE et non plus par
    /// MVV-LVA — prises see >= 0 (200 000 + see) AVANT les killers, prises
    /// see < 0 (-1 000 000 + see) TOUT EN BAS, sous l'historique : une prise
    /// perdante est un espoir plus maigre qu'un bon coup calme.
    fn cle_ordre(&self, pos: &Chess, m: &Move, coup_tt: u16, ply: u32, couleur: Color) -> i32 {
        let c = compacter(m);
        if coup_tt != COUP_AUCUN && c == coup_tt {
            return 1_000_000;
        }
        if !self.mode_classique && m.is_capture() {
            let s = see(pos, m);
            return if s >= 0 { 200_000 + s } else { -1_000_000 + s };
        }
        let cle = cle_mvv_lva(m);
        if cle != 0 {
            // (classique) prise et/ou promotion ; (raffiné) promotion calme
            return cle;
        }
        let k = &self.killers[ply as usize];
        if c == k[0] {
            return 80_000;
        }
        if c == k[1] {
            return 79_000;
        }
        self.historique[Self::idx_historique(couleur, m)].min(60_000) as i32
    }

    fn idx_historique(couleur: Color, m: &Move) -> usize {
        let from = m.from().map_or(0, usize::from);
        let to = usize::from(m.to());
        idx_couleur(couleur) * 64 * 64 + from * 64 + to
    }

    fn note_killer(&mut self, ply: u32, coup: u16) {
        let k = &mut self.killers[ply as usize];
        if k[0] != coup {
            k[1] = k[0];
            k[0] = coup;
        }
    }

    /// Historique : +profondeur² à la coupure bêta. En cas de débordement
    /// (rarissime), toute la table est divisée par 2 — les ordres relatifs
    /// sont conservés.
    fn note_historique(&mut self, couleur: Color, m: &Move, profondeur: u32) {
        let idx = Self::idx_historique(couleur, m);
        self.historique[idx] += profondeur * profondeur;
        if self.historique[idx] > 1_000_000 {
            for h in &mut self.historique {
                *h /= 2;
            }
        }
    }
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{Bot, NetBot};
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::{Rng, SeedableRng};
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Square};
    use std::collections::HashMap;

    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    /// Réseau réduit DÉTERMINISTE dérivé de Mlp::new(0) : mêmes 773 entrées,
    /// mais 12 neurones cachés (troncature des poids du réseau neuf).
    ///
    /// POURQUOI : en profil dev (cargo test --lib, sans optimisations), un
    /// forward du réseau complet 773→512→64→1 coûte ~7 ms ; une recherche de
    /// 3 000 nœuds en ferait ~2 000 → l'arène du test (4) durerait des heures.
    /// Le réseau réduit (~13 000 multiplications) garde tout ce qui compte
    /// pour ces tests : une évaluation non triviale, déterministe, identique
    /// pour les deux camps. Le réseau complet reste couvert par le test de
    /// fumée `reseau_complet_fumee`.
    fn reseau_reduit() -> Arc<Mlp> {
        let base = Mlp::new(0);
        let (n_in, cache) = (N_FEATURES, 12usize);
        let w0: Vec<f32> = base.weights[0][..cache * n_in].to_vec();
        let b0: Vec<f32> = base.biases[0][..cache].to_vec();
        let w1: Vec<f32> = base.weights[1][..cache].to_vec();
        let b1: Vec<f32> = vec![base.biases[2][0]];
        let formes_w = [w0.len(), w1.len()];
        let formes_b = [b0.len(), b1.len()];
        Arc::new(Mlp {
            sizes: vec![n_in, cache, 1],
            weights: vec![w0, w1],
            biases: vec![b0, b1],
            adam_mw: formes_w.iter().map(|&n| vec![0.0; n]).collect(),
            adam_vw: formes_w.iter().map(|&n| vec![0.0; n]).collect(),
            adam_mb: formes_b.iter().map(|&n| vec![0.0; n]).collect(),
            adam_vb: formes_b.iter().map(|&n| vec![0.0; n]).collect(),
            steps: 0,
            pas_colonnes: vec![0u64; n_in],
        })
    }

    /// Réseau linéaire [773 → 1] aux poids MATÉRIELS (nos P,N,B,R,Q positifs,
    /// les leurs négatifs, roques/en-passant à 0) : une évaluation jouable,
    /// déterministe et quasi gratuite en profil dev. L'arène du test (4)
    /// l'utilise pour les DEUX camps : le duel mesure alors l'apport de la
    /// RECHERCHE (tactique profonde, mats exacts) à savoir égal — un réseau
    /// purement aléatoire ne donne qu'un effet Beal marginal (~70 % mesuré),
    /// trop juste pour servir de garde-fou fiable ; le réseau NEUF reste
    /// couvert par les autres tests (mats, déterminisme, TT, fumée).
    fn reseau_materiel() -> Arc<Mlp> {
        let valeurs = [0.10f32, 0.30, 0.32, 0.50, 0.90, 0.0];
        let mut w0 = vec![0.0f32; N_FEATURES];
        for (plan, &v) in valeurs.iter().enumerate() {
            for case in 0..64 {
                w0[plan * 64 + case] = v; // nos pièces
                w0[(6 + plan) * 64 + case] = -v; // les leurs
            }
        }
        Arc::new(Mlp {
            sizes: vec![N_FEATURES, 1],
            weights: vec![w0],
            biases: vec![vec![0.0]],
            adam_mw: vec![vec![0.0; N_FEATURES]],
            adam_vw: vec![vec![0.0; N_FEATURES]],
            adam_mb: vec![vec![0.0]],
            adam_vb: vec![vec![0.0]],
            steps: 0,
            pas_colonnes: vec![0u64; N_FEATURES],
        })
    }

    /// Réseau matériel BRUITÉ : les poids de reseau_materiel() plus un petit
    /// bruit positionnel déterministe (±0.008, LCG sur l'indice de feature).
    /// L'évaluation reste linéaire (quasi gratuite en profil dev) et dominée
    /// par le matériel, mais DISCRIMINE : avec le réseau matériel pur, presque
    /// toutes les lignes calmes s'annulent à 0.0 exactement — arbres
    /// squelettiques et fenêtres dégénérées, rien à mesurer. Le bruit imite
    /// une composante positionnelle : arbres réalistes pour les A/B.
    fn reseau_materiel_bruite() -> Arc<Mlp> {
        reseau_materiel_bruite_amp(0.016)
    }

    /// Variante à amplitude de bruit réglable (crête à crête). `plans` : seuls
    /// ces plans de pièces (0-5 nous, 6-11 eux) reçoivent du bruit — un bruit
    /// PARCIMONIEUX préserve les égalités exactes des lignes calmes qui ne
    /// touchent pas ces pièces, donc les coupures immédiates de la quiescence
    /// (arbres compacts), tout en donnant une direction de jeu.
    fn reseau_materiel_bruite_plans(amplitude: f32, plans: &[usize]) -> Arc<Mlp> {
        let mut net = Arc::try_unwrap(reseau_materiel()).ok().expect("Arc unique");
        let mut etat = 0x9E37_79B9_7F4A_7C15u64;
        for (i, w) in net.weights[0].iter_mut().enumerate() {
            // LCG 64 bits (constantes de Knuth) : déterministe, sans rand.
            etat = etat
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if plans.contains(&(i / 64)) {
                *w += ((etat >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * amplitude;
            }
        }
        Arc::new(net)
    }

    fn reseau_materiel_bruite_amp(amplitude: f32) -> Arc<Mlp> {
        reseau_materiel_bruite_plans(amplitude, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    }

    fn limites_prof(p: u32) -> Limites {
        Limites { max_noeuds: 0, max_profondeur: p, movetime_ms: 0 }
    }

    fn limites_noeuds(n: u64) -> Limites {
        Limites { max_noeuds: n, max_profondeur: 0, movetime_ms: 0 }
    }

    /// (1) Mat en 1 : tour a1, roi noir g8 enfermé par ses pions → Ra8#.
    #[test]
    fn mat_en_1_trouve() {
        let pos = pos_de_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
        let mut r = Recherche::new(reseau_reduit(), 14);
        let res = r.cherche(&pos, limites_prof(2));
        let coup = res.coup.expect("un coup légal existe");
        assert!(res.score > 900.0, "score de mat attendu, obtenu {}", res.score);
        assert_eq!(coup.to(), Square::A8, "Ra8# attendu, obtenu {coup:?}");
        // Mat en 1 exactement : SCORE_MAT - 1.
        assert!((res.score - (SCORE_MAT - 1.0)).abs() < 1e-3);
    }

    /// (2) Mat en 2 (mat du couloir à deux tours) trouvé à profondeur 4 :
    /// tours a2 et b1 contre roi h8 (ex. 1.Ra7 Kg8 2.Rb8#, plusieurs échelles
    /// gagnent). Aucun mat en 1 n'existe : le score doit valoir exactement
    /// SCORE_MAT - 3 (mat au 3e demi-coup), et le coup choisi doit réellement
    /// forcer le mat : la position fille, cherchée à son tour, vaut
    /// -(SCORE_MAT - 2) pour le camp maté.
    #[test]
    fn mat_en_2_trouve_profondeur_4() {
        let pos = pos_de_fen("7k/8/8/8/8/8/R7/1R5K w - - 0 1");
        let mut r = Recherche::new(reseau_reduit(), 16);
        let res = r.cherche(&pos, limites_prof(4));
        let coup = res.coup.expect("un coup légal existe");
        assert!(
            (res.score - (SCORE_MAT - 3.0)).abs() < 1e-3,
            "mat en 2 (score {}) attendu, obtenu {}",
            SCORE_MAT - 3.0,
            res.score
        );
        assert!(res.profondeur >= 3, "profondeur {} < 3", res.profondeur);
        // Contre-vérification : après le coup choisi, le camp adverse est bien
        // maté en 2 demi-coups quoi qu'il joue.
        let fille = pos.play(&coup).expect("coup légal");
        let mut rv = Recherche::new(reseau_reduit(), 14);
        let verdict = rv.cherche(&fille, limites_prof(3));
        assert!(
            (verdict.score + (SCORE_MAT - 2.0)).abs() < 1e-3,
            "le coup {coup:?} ne force pas le mat : verdict {}",
            verdict.score
        );
    }

    /// (3) Déterminisme : mêmes limites de nœuds, deux chercheurs neufs →
    /// résultats identiques bit à bit (aucune horloge en jeu).
    #[test]
    fn deterministe_a_noeuds_fixes() {
        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let net = reseau_reduit();
        let mut r1 = Recherche::new(net.clone(), 14);
        let mut r2 = Recherche::new(net, 14);
        let a = r1.cherche(&pos, limites_noeuds(1500));
        let b = r2.cherche(&pos, limites_noeuds(1500));
        assert_eq!(a.coup, b.coup);
        assert_eq!(a.score, b.score);
        assert_eq!(a.profondeur, b.profondeur);
        assert_eq!(a.noeuds, b.noeuds);
        assert_eq!(a.scores_racine.len(), b.scores_racine.len());
        for ((ma, va), (mb, vb)) in a.scores_racine.iter().zip(&b.scores_racine) {
            assert_eq!(ma, mb);
            assert_eq!(va, vb);
        }
    }

    /// Joue une partie chercheur (3000 nœuds) contre NetBot 1 pli, même réseau.
    /// `ouverture` : plis initiaux joués au hasard (les deux bots étant quasi
    /// déterministes, c'est elle qui diversifie les parties). Renvoie le
    /// résultat côté chercheur : 1 victoire, 0.5 nulle, 0 défaite.
    fn partie_recherche_contre_1_pli(
        recherche: &mut Recherche,
        net: &Mlp,
        chercheur_blanc: bool,
        graine: u64,
        ouverture: &[Move],
    ) -> (f32, u32) {
        let mut pos = Chess::default();
        let mut repetitions: HashMap<u64, u8> = HashMap::new();
        repetitions.insert(zobrist(&pos), 1);
        for m in ouverture {
            pos = pos.play(m).expect("coup d'ouverture légal");
            *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
        }
        recherche.nouvelle_partie();
        let mut adversaire = NetBot::new(net, graine, 0.0, 1);
        let limites = limites_noeuds(3000);
        let mut plies = 0u32;

        let resultat_blancs = loop {
            let coups = pos.legal_moves();
            if coups.is_empty() {
                break if pos.is_check() {
                    if pos.turn() == Color::White { -1.0 } else { 1.0 }
                } else {
                    0.0
                };
            }
            if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= 200 {
                break 0.0;
            }
            let tour_chercheur = (pos.turn() == Color::White) == chercheur_blanc;
            let m = if tour_chercheur {
                recherche.cherche(&pos, limites).coup.expect("coup légal")
            } else {
                adversaire.choose(&pos).expect("coup légal")
            };
            pos = pos.play(&m).expect("coup légal");
            plies += 1;
            let c = repetitions.entry(zobrist(&pos)).or_insert(0);
            *c += 1;
            if *c >= 3 {
                break 0.0;
            }
        };
        let cote = if chercheur_blanc { resultat_blancs } else { -resultat_blancs };
        ((cote + 1.0) / 2.0, plies)
    }

    /// (4) La recherche à 3000 nœuds bat le réseau brut (1 pli, MÊME réseau
    /// pour les deux camps) : mini-arène de 10 parties, ouvertures aléatoires
    /// appariées (couleurs échangées), score exigé >= 70 %. Voir
    /// reseau_materiel() pour le choix du réseau de duel.
    #[test]
    fn recherche_bat_reseau_brut_mini_arene() {
        let net = reseau_materiel();
        let mut recherche = Recherche::new(net.clone(), 16);
        let mut points = 0.0f32;
        for paire in 0..5u64 {
            // Ouverture aléatoire de 4 plis, partagée par les deux parties de
            // la paire (équité : chaque camp joue la même ouverture des deux
            // couleurs).
            let mut rng = StdRng::seed_from_u64(0xA5E5 + paire);
            let mut pos = Chess::default();
            let mut ouverture = Vec::new();
            for _ in 0..4 {
                let m = pos
                    .legal_moves()
                    .choose(&mut rng)
                    .cloned()
                    .expect("ouverture jouable");
                pos = pos.play(&m).expect("coup légal");
                ouverture.push(m);
            }
            for (i, chercheur_blanc) in [(0u64, true), (1u64, false)] {
                let (pts, plies) = partie_recherche_contre_1_pli(
                    &mut recherche,
                    &net,
                    chercheur_blanc,
                    7000 + paire * 2 + i,
                    &ouverture,
                );
                println!(
                    "  partie {} (chercheur {}) : {} en {} plis",
                    paire * 2 + i,
                    if chercheur_blanc { "blanc" } else { "noir" },
                    pts,
                    plies
                );
                points += pts;
            }
        }
        let score = points / 10.0;
        println!("mini-arène recherche(3000 nœuds) vs 1 pli : score {score}");
        assert!(
            score >= 0.7,
            "la recherche ne domine pas le réseau brut : {score} < 0.70"
        );
    }

    /// (5) TT persistante : rechercher deux fois la même position sur le même
    /// chercheur → la seconde passe coûte beaucoup moins de nœuds. (Le coup
    /// peut légitimement différer : réutiliser des BORNES mises en cache sous
    /// d'autres fenêtres rend la recherche « instable » entre coups quasi à
    /// égalité — comportement standard de tous les moteurs alpha-bêta à TT.)
    #[test]
    fn tt_reduit_les_noeuds_en_recherche_repetee() {
        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let mut r = Recherche::new(reseau_reduit(), 16);
        let a = r.cherche(&pos, limites_prof(4));
        let b = r.cherche(&pos, limites_prof(4));
        assert!(a.coup.is_some() && b.coup.is_some());
        // Position calme : les deux scores restent des valeurs réseau.
        assert!(a.score.abs() < 1.0 && b.score.abs() < 1.0);
        assert!(
            b.noeuds * 2 <= a.noeuds,
            "2e recherche pas assez accélérée par la TT : {} vs {}",
            b.noeuds,
            a.noeuds
        );
    }

    /// (6) Parité NNUE / forward complet : même réseau (biais rendus non nuls
    /// pour couvrir leur transport dans la pile d'accumulateurs), même
    /// position, même graine, mêmes limites de nœuds → MÊME coup et MÊME
    /// score (à 1e-4). Le chercheur de référence est forcé sur le chemin de
    /// secours (encode + forward_one) via force_forward_complet() —
    /// exactement le code d'avant l'intégration NNUE.
    #[test]
    fn nnue_meme_coup_et_score_que_forward_complet() {
        let mut net = Arc::try_unwrap(reseau_reduit()).ok().expect("Arc unique");
        let mut rng = StdRng::seed_from_u64(0xB1A15);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        let net = Arc::new(net);

        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let mut nnue = Recherche::new(net.clone(), 14);
        let mut reference = Recherche::new(net, 14);
        reference.force_forward_complet();

        let a = nnue.cherche(&pos, limites_noeuds(4000));
        let b = reference.cherche(&pos, limites_noeuds(4000));

        assert_eq!(a.coup, b.coup, "coup NNUE != coup forward complet");
        assert!(
            (a.score - b.score).abs() <= 1e-4,
            "scores divergents : NNUE {} vs forward complet {}",
            a.score,
            b.score
        );
        assert_eq!(a.profondeur, b.profondeur);
        // Les scores racine de la dernière itération complète coïncident
        // aussi coup par coup (même arbre, mêmes bornes fail-soft).
        assert_eq!(a.scores_racine.len(), b.scores_racine.len());
        for ((ma, va), (mb, vb)) in a.scores_racine.iter().zip(&b.scores_racine) {
            assert_eq!(ma, mb);
            assert!((va - vb).abs() <= 1e-4, "{ma:?} : {va} vs {vb}");
        }
    }

    /// Bench NNUE (ignoré par défaut) : nœuds/s et profondeur atteinte en
    /// 150 ms sur la position initiale, réseau complet [773,512,64,1] —
    /// AVANT (forward complet forcé) puis APRÈS (évaluation incrémentale).
    /// Lancer : `cargo test --lib search:: -- --ignored --nocapture`
    /// (chiffres représentatifs en --release uniquement).
    #[test]
    #[ignore]
    fn bench_nnue_avant_apres_150ms() {
        let net = Arc::new(Mlp::new(0));
        for (nom, forcer_forward) in
            [("AVANT (encode + forward complet)", true), ("APRÈS (NNUE incrémental)   ", false)]
        {
            let mut r = Recherche::new(net.clone(), 20);
            if forcer_forward {
                r.force_forward_complet();
            }
            let debut = Instant::now();
            let res = r.cherche(
                &Chess::default(),
                Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: 150 },
            );
            let d = debut.elapsed();
            println!(
                "{nom} : profondeur {:>2}, {:>8} noeuds en {:?} ({:.0} noeuds/s)",
                res.profondeur,
                res.noeuds,
                d,
                res.noeuds as f64 / d.as_secs_f64()
            );
        }
    }

    /// Fumée avec le réseau COMPLET (Mlp::new(0)) : la recherche s'intègre au
    /// vrai réseau, rend un coup légal et un score réseau borné.
    #[test]
    fn reseau_complet_fumee() {
        let net = Arc::new(Mlp::new(0));
        let mut r = Recherche::new(net, 12);
        let res = r.cherche(&Chess::default(), limites_noeuds(60));
        let coup = res.coup.expect("coup légal en position initiale");
        assert!(Chess::default().is_legal(&coup));
        assert!(res.score.is_finite() && res.score.abs() < 1.0);
        assert!(res.profondeur >= 1);
        assert!(res.noeuds >= 20 && res.noeuds <= 200, "noeuds = {}", res.noeuds);
        assert_eq!(res.scores_racine.len(), 20);
    }

    /// Position terminale : aucun coup → coup None et score de mat/pat.
    #[test]
    fn position_matee_rend_none() {
        // Mat du couloir déjà consommé : trait aux noirs, matés.
        let pos = pos_de_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1");
        let mut r = Recherche::new(reseau_reduit(), 10);
        let res = r.cherche(&pos, limites_prof(3));
        assert!(res.coup.is_none());
        assert_eq!(res.score, -SCORE_MAT);
        assert!(res.scores_racine.is_empty());
    }

    /// Trouve le coup légal from→to (échoue s'il n'existe pas).
    fn coup(pos: &Chess, de: Square, vers: Square) -> Move {
        pos.legal_moves()
            .iter()
            .find(|m| m.from() == Some(de) && m.to() == vers)
            .cloned()
            .unwrap_or_else(|| panic!("coup {de}-{vers} illégal ici"))
    }

    /// (7) SEE, cas écrits à la main : les quatre archétypes du cahier des
    /// charges, valeurs vérifiées au tableau noir avec l'algorithme d'échange.
    #[test]
    fn see_cas_unitaires() {
        // (a) Prise gagnante simple : exd5 gagne la dame, personne ne défend.
        let pos = pos_de_fen("k7/8/8/3q4/4P3/8/8/K7 w - - 0 1");
        assert_eq!(see(&pos, &coup(&pos, Square::E4, Square::D5)), 900);

        // (b) Échange perdant : Dxd5 prend un pion défendu par le pion c6 —
        // +100 (pion) - 900 (dame reprise) = -800.
        let pos = pos_de_fen("k7/8/2p5/3p4/8/8/3Q4/K7 w - - 0 1");
        assert_eq!(see(&pos, &coup(&pos, Square::D2, Square::D5)), -800);

        // (c) Rayon X derrière une tour : Txd5 (pion défendu par Td8), mais la
        // Td1 soutient À TRAVERS la tour d3 partie prendre —
        // +100 - 500 + 500 = +100. Sans rayons X, le SEE rendrait -400.
        let pos = pos_de_fen("k2r4/8/8/3p4/8/3R4/8/K2R4 w - - 0 1");
        assert_eq!(see(&pos, &coup(&pos, Square::D3, Square::D5)), 100);

        // (d) Prise égale : exd5 pion contre pion, d5 défendu par c6 —
        // +100 - 100 = 0 : l'échange ne rapporte rien mais ne coûte rien.
        let pos = pos_de_fen("k7/8/2p5/3p4/4P3/8/8/K7 w - - 0 1");
        assert_eq!(see(&pos, &coup(&pos, Square::E4, Square::D5)), 0);
    }

    /// (7 bis) Régression (audit « le mat est vu ») : une prise qui MATE mais
    /// que le SEE juge perdante ne doit PAS être élaguée par le filtre de
    /// quiescence du mode raffiné. Ici Qxh7# : h7 est « défendu » par le Cg5
    /// — cloué de manière absolue par la Tg1, mais le SEE ignore les clouages
    /// et compte sa reprise (see = -800 < 0) — et Rxh7 est interdit par le
    /// soutien du Fc2 ; le défenseur compté par le SEE ne reprendra jamais
    /// puisque la partie est finie. Les DEUX autres prises blanches (Txg5,
    /// Fxh7) sont elles aussi filtrées (see < 0) : sans le repêchage des
    /// prises qui donnent échec, la liste tactique était VIDE et la
    /// quiescence rendait le stand-pat réseau (borné) au lieu du mat.
    #[test]
    fn quiescence_voit_le_mat_dune_prise_see_negative() {
        let pos = pos_de_fen("5rk1/1Q5p/7p/6n1/8/8/2B5/K5R1 w - - 0 1");
        // Pré-conditions du scénario : la prise a un SEE négatif ET mate.
        let qxh7 = coup(&pos, Square::B7, Square::H7);
        assert!(see(&pos, &qxh7) < 0, "le scénario exige see(Qxh7) < 0");
        let apres = pos.clone().play(&qxh7).expect("Qxh7 légal");
        assert!(
            apres.legal_moves().is_empty() && apres.is_check(),
            "le scénario exige que Qxh7 soit mat"
        );
        // Feuille de quiescence en mode raffiné (filtre SEE actif, ply 0,
        // fenêtre pleine) : le mat doit être vu.
        let mut r = Recherche::new(reseau_reduit(), 12);
        let v = r.quiesce(&pos, 0, PROF_QUIESCENCE, f32::NEG_INFINITY, f32::INFINITY);
        assert!(
            v > SEUIL_MAT,
            "mat invisible à la feuille de quiescence : {v} (attendu > {SEUIL_MAT})"
        );
    }

    /// (8) Le mode classique (raffinements débrayés) reste la recherche
    /// d'origine : mêmes mats exacts, mêmes scores. (Les tests (1) et (2)
    /// couvrent déjà les mêmes mats AVEC raffinements — mode par défaut.)
    #[test]
    fn mode_classique_mats_intacts() {
        for (fen, prof, attendu) in [
            ("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", 2, SCORE_MAT - 1.0),
            ("7k/8/8/8/8/8/R7/1R5K w - - 0 1", 4, SCORE_MAT - 3.0),
        ] {
            let mut r = Recherche::new(reseau_reduit(), 14);
            r.mode_classique = true;
            let res = r.cherche(&pos_de_fen(fen), limites_prof(prof));
            assert!(
                (res.score - attendu).abs() < 1e-3,
                "{fen} : score {} attendu, obtenu {}",
                attendu,
                res.score
            );
        }
    }

    /// (9) Réduction mesurée : à profondeur 6 égale, la recherche raffinée
    /// (LMR + aspiration + SEE) dépense au plus 60 % des nœuds de la
    /// classique — position initiale et deux milieux de partie (Kiwipete et
    /// la « position 5 » des suites perft : légales et bien connues).
    /// Réseau matériel bruité : voir reseau_materiel_bruite() — il faut une
    /// évaluation qui discrimine pour mesurer quoi que ce soit.
    #[test]
    fn raffinements_reduisent_les_noeuds_profondeur_6() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];
        let net = reseau_materiel_bruite();
        let (mut tot_c, mut tot_r) = (0u64, 0u64);
        for fen in fens {
            let pos = pos_de_fen(fen);
            let mut classique = Recherche::new(net.clone(), 18);
            classique.mode_classique = true;
            let nc = classique.cherche(&pos, limites_prof(6)).noeuds;
            let mut raffinee = Recherche::new(net.clone(), 18);
            let nr = raffinee.cherche(&pos, limites_prof(6)).noeuds;
            println!(
                "  {fen}\n    classique {nc} nœuds, raffinée {nr} nœuds ({:.1} %)",
                100.0 * nr as f64 / nc as f64
            );
            tot_c += nc;
            tot_r += nr;
            assert!(
                nr * 10 <= nc * 6,
                "raffinée {nr} > 60 % de classique {nc} sur {fen}"
            );
        }
        println!(
            "  TOTAL : classique {tot_c}, raffinée {tot_r} ({:.1} %)",
            100.0 * tot_r as f64 / tot_c as f64
        );
    }

    /// Une partie chercheur A contre chercheur B au même budget de nœuds.
    /// Renvoie le point côté A (1 victoire, 0.5 nulle, 0 défaite) et les plis.
    fn partie_deux_chercheurs(
        a: &mut Recherche,
        b: &mut Recherche,
        a_blanc: bool,
        ouverture: &[Move],
        budget: u64,
    ) -> (f32, u32) {
        let mut pos = Chess::default();
        let mut repetitions: HashMap<u64, u8> = HashMap::new();
        repetitions.insert(zobrist(&pos), 1);
        for m in ouverture {
            pos = pos.play(m).expect("coup d'ouverture légal");
            *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
        }
        a.nouvelle_partie();
        b.nouvelle_partie();
        let limites = limites_noeuds(budget);
        let mut plies = 0u32;
        let resultat_blancs = loop {
            let coups = pos.legal_moves();
            if coups.is_empty() {
                break if pos.is_check() {
                    if pos.turn() == Color::White { -1.0 } else { 1.0 }
                } else {
                    0.0
                };
            }
            if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= 200 {
                break 0.0;
            }
            let tour_a = (pos.turn() == Color::White) == a_blanc;
            let joueur = if tour_a { &mut *a } else { &mut *b };
            let m = joueur.cherche(&pos, limites).coup.expect("coup légal");
            pos = pos.play(&m).expect("coup légal");
            plies += 1;
            let c = repetitions.entry(zobrist(&pos)).or_insert(0);
            *c += 1;
            if *c >= 3 {
                break 0.0;
            }
        };
        let cote = if a_blanc { resultat_blancs } else { -resultat_blancs };
        ((cote + 1.0) / 2.0, plies)
    }

    /// (10) A/B d'arène : raffiné vs classique, 20 parties à ouvertures
    /// appariées (couleurs échangées), MÊME budget de 3000 nœuds et MÊME
    /// réseau (jamais entraîné) pour les deux camps : à savoir égal, le duel
    /// mesure l'apport des RAFFINEMENTS (profondeur effective à budget égal).
    /// Réseau matériel à bruit PARCIMONIEUX (pièces mineures seulement) :
    /// - matériel pur, quasi toutes les lignes calmes valent 0.0 exactement —
    ///   13 nulles sur 20 mesurées, la profondeur supplémentaire n'a RIEN à
    ///   encaisser (et le raffiné, à profondeur égale sur ces arbres déjà
    ///   squelettiques, ne paie que le surcoût de ses sondages) ;
    /// - bruit sur TOUTES les cases, plus une seule égalité exacte : les
    ///   coupures immédiates de la quiescence disparaissent, les arbres
    ///   décuplent et 3000 nœuds ne dépassent plus la profondeur 2-3, où LMR
    ///   (prof >= 3) et aspiration (itération >= 3) s'engagent à peine ;
    /// - bruit sur les seuls plans des mineures : les lignes calmes qui ne
    ///   les touchent pas gardent leurs égalités (arbres compacts, profondeur
    ///   4-5 au budget) ET le jeu a une direction — mesuré : +1 pli pour le
    ///   raffiné sur la moitié des positions à 3000 nœuds.
    /// Attendu >= 55 % ; l'assertion est à 50 % pour la marge de bruit et le
    /// score EXACT est imprimé.
    #[test]
    fn arene_ab_raffine_bat_classique() {
        let net = reseau_materiel_bruite_plans(0.016, &[1, 2, 7, 8]);
        let mut raffine = Recherche::new(net.clone(), 16);
        let mut classique = Recherche::new(net.clone(), 16);
        classique.mode_classique = true;
        let mut points = 0.0f32;
        for paire in 0..10u64 {
            // Ouverture aléatoire de 4 plis, partagée par les deux parties de
            // la paire (équité : chaque camp la joue des deux couleurs).
            let mut rng = StdRng::seed_from_u64(0xAB0 + paire);
            let mut pos = Chess::default();
            let mut ouverture = Vec::new();
            for _ in 0..4 {
                let m = pos
                    .legal_moves()
                    .choose(&mut rng)
                    .cloned()
                    .expect("ouverture jouable");
                pos = pos.play(&m).expect("coup légal");
                ouverture.push(m);
            }
            for a_blanc in [true, false] {
                let (pts, plies) = partie_deux_chercheurs(
                    &mut raffine,
                    &mut classique,
                    a_blanc,
                    &ouverture,
                    3000,
                );
                println!(
                    "  paire {paire}, raffiné {} : {pts} en {plies} plis",
                    if a_blanc { "blanc" } else { "noir" }
                );
                points += pts;
            }
        }
        let score = points / 20.0;
        println!("A/B raffiné vs classique (3000 nœuds/coup) : score raffiné = {score}");
        assert!(
            score >= 0.5,
            "le raffiné ne domine pas le classique : {score} < 0.50"
        );
    }

    /// Diagnostic (ignoré par défaut) : profondeur complète atteinte à 3000
    /// nœuds, classique vs raffiné, réseau matériel bruité — vérifie que les
    /// économies de nœuds se convertissent bien en profondeur au budget de
    /// l'arène A/B.
    #[test]
    #[ignore]
    fn diag_profondeur_a_budget_3000() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];
        for (nom, net) in [
            ("matériel pur", reseau_materiel()),
            ("bruit mineurs 0.016", reseau_materiel_bruite_plans(0.016, &[1, 2, 7, 8])),
            ("bruit cavaliers 0.016", reseau_materiel_bruite_plans(0.016, &[1, 7])),
            ("bruit rois 0.016", reseau_materiel_bruite_plans(0.016, &[5, 11])),
            ("bruit total 0.016", reseau_materiel_bruite_amp(0.016)),
        ] {
            println!("=== {nom} ===");
            for fen in fens {
                let pos = pos_de_fen(fen);
                let mut c = Recherche::new(net.clone(), 16);
                c.mode_classique = true;
                let rc = c.cherche(&pos, limites_noeuds(3000));
                let mut r = Recherche::new(net.clone(), 16);
                let rr = r.cherche(&pos, limites_noeuds(3000));
                println!(
                    "  classique : prof {} | raffiné : prof {}  ({fen})",
                    rc.profondeur, rr.profondeur
                );
            }
        }
    }

    // --- Harnais de parité mono-thread (chantier lazy SMP) -------------------

    /// Réseau réduit à biais non nuls (même recette que le test de parité
    /// NNUE) : couvre le transport des biais dans les piles d'accumulateurs.
    fn reseau_reduit_biaise() -> Arc<Mlp> {
        let mut net = Arc::try_unwrap(reseau_reduit()).ok().expect("Arc unique");
        let mut rng = StdRng::seed_from_u64(0xB1A15);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        Arc::new(net)
    }

    /// `n` positions variées et déterministes : marches aléatoires (graines
    /// fixes) de 2 à 41 plis depuis la position initiale, jamais terminales.
    /// NE PAS MODIFIER cette génération : la référence de parité en dépend.
    fn positions_variees(n: usize) -> Vec<Chess> {
        let mut v = Vec::new();
        let mut graine = 0u64;
        while v.len() < n {
            graine += 1;
            let mut rng = StdRng::seed_from_u64(0x5EED_0000 + graine);
            let mut pos = Chess::default();
            for _ in 0..(2 + (graine as usize * 7) % 40) {
                let coups = pos.legal_moves();
                if coups.is_empty() || pos.is_insufficient_material() || pos.halfmoves() >= 100 {
                    break;
                }
                pos = pos
                    .play(coups.choose(&mut rng).expect("liste non vide"))
                    .expect("coup légal");
            }
            if !pos.legal_moves().is_empty() {
                v.push(pos);
            }
        }
        v
    }

    /// Empreinte comportementale du chercheur mono-thread : pour chaque
    /// (config, position), le coup choisi, le score AU BIT PRÈS, la profondeur
    /// et le nombre EXACT de nœuds, à budget de nœuds fixe (aucune horloge en
    /// jeu, tout est déterministe). La TT est conservée d'une position à
    /// l'autre au sein d'une config : la politique de remplacement fait
    /// partie du comportement capturé.
    fn empreinte_parite() -> Vec<String> {
        let net = reseau_reduit_biaise();
        let positions = positions_variees(200);
        let mut lignes = Vec::new();
        for (nom, classique, int8) in
            [("defaut", false, false), ("classique", true, false), ("int8", false, true)]
        {
            let mut r = Recherche::new(net.clone(), 18);
            r.mode_classique = classique;
            r.utilise_int8 = int8;
            for pos in &positions {
                let res = r.cherche(pos, limites_noeuds(2500));
                let coup = res
                    .coup
                    .map(|m| m.to_uci(CastlingMode::Standard).to_string())
                    .unwrap_or_else(|| "aucun".into());
                lignes.push(format!(
                    "{nom};{};{coup};{:08x};{};{}",
                    Fen::from_position(pos.clone(), EnPassantMode::Legal),
                    res.score.to_bits(),
                    res.profondeur,
                    res.noeuds
                ));
            }
        }
        lignes
    }

    /// (Parité SMP) Test piloté par ECHEC_PARITE_REF :
    /// - le fichier n'existe pas → l'empreinte du code COURANT y est écrite ;
    /// - il existe → l'empreinte recalculée doit lui être IDENTIQUE.
    /// Généré AVANT le chantier SMP, rejoué APRÈS : threads=1 doit rejouer
    /// exactement les mêmes coups, scores (au bit), profondeurs et nœuds.
    /// Lancer : ECHEC_PARITE_REF=<fichier> cargo test --lib --release
    ///          parite_threads_1 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn parite_threads_1_contre_reference() {
        let Some(chemin) = std::env::var_os("ECHEC_PARITE_REF") else {
            println!("ECHEC_PARITE_REF absent : rien à faire");
            return;
        };
        let chemin = std::path::PathBuf::from(chemin);
        let lignes = empreinte_parite();
        if chemin.exists() {
            let attendu = std::fs::read_to_string(&chemin).expect("lecture de la référence");
            let attendu: Vec<&str> = attendu.lines().collect();
            let mut ecarts = 0usize;
            for (i, (a, b)) in attendu.iter().zip(&lignes).enumerate() {
                if *a != b.as_str() {
                    ecarts += 1;
                    if ecarts <= 10 {
                        println!("écart ligne {i} :\n  réf : {a}\n  ici : {b}");
                    }
                }
            }
            assert_eq!(attendu.len(), lignes.len(), "nombre de lignes");
            assert_eq!(ecarts, 0, "{ecarts} écart(s) contre la référence");
            println!("parité stricte : {} lignes identiques", lignes.len());
        } else {
            std::fs::write(&chemin, lignes.join("\n")).expect("écriture de la référence");
            println!("référence écrite ({} lignes) : {}", lignes.len(), chemin.display());
        }
    }

    /// Compactage TT : aller-retour exact champ à champ (score au bit près,
    /// scores de mat compris), et détection d'une écriture déchirée par la
    /// validation XOR du hachage lockless.
    #[test]
    fn tt_paquette_depaquette_exacts() {
        for (score, coup, profondeur, drapeau) in [
            (0.0f32, COUP_AUCUN, 0u8, DRAPEAU_VIDE),
            (0.123_456_79, 4095u16, 17, DRAPEAU_EXACT),
            (-0.987_654_3, 513, 255, DRAPEAU_BORNE_INF),
            (SCORE_MAT - 7.0, u16::MAX, 64, DRAPEAU_BORNE_SUP),
            (-(SCORE_MAT - 12.0), 1, 1, DRAPEAU_EXACT),
        ] {
            let d = paquette(EntreeTT { score, coup, profondeur, drapeau });
            let r = depaquette(d);
            assert_eq!(r.score.to_bits(), score.to_bits());
            assert_eq!(r.coup, coup);
            assert_eq!(r.profondeur, profondeur);
            assert_eq!(r.drapeau, drapeau);
        }
        // Écriture déchirée simulée : cle_x d'une entrée, donnees d'une autre
        // → la clé reconstituée ne colle plus, l'entrée est ignorée.
        let e1 = paquette(EntreeTT { score: 0.25, coup: 100, profondeur: 9, drapeau: DRAPEAU_EXACT });
        let e2 = paquette(EntreeTT { score: -0.5, coup: 7, profondeur: 3, drapeau: DRAPEAU_BORNE_INF });
        let (cle1, cle2) = (0xDEAD_BEEF_1234_5678u64, 0x0BAD_F00D_8765_4321u64);
        assert_eq!((cle1 ^ e1) ^ e1, cle1); // écriture propre : validée
        assert_ne!((cle1 ^ e1) ^ e2, cle1); // entrelacement : détecté...
        assert_ne!((cle1 ^ e1) ^ e2, cle2); // ...pour les deux clés en course
    }

    /// (SMP) Fumée 2 threads : un coup légal et un score fini à 150 ms sur un
    /// milieu de partie, TT partagée effectivement sondée. La couverture SMP
    /// sérieuse vit dans smp_8_threads_50_positions_500ms (ignoré, release).
    #[test]
    fn smp_2_threads_fumee() {
        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let mut r = Recherche::new(reseau_reduit_biaise(), 16);
        r.threads = 2;
        let res =
            r.cherche(&pos, Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: 150 });
        let coup = res.coup.expect("coup légal");
        assert!(pos.is_legal(&coup));
        assert!(res.score.is_finite());
        assert!(r.tt_sondes > 0, "la TT partagée n'a jamais été sondée");
    }

    /// (SMP, ignoré par défaut) Le test du contrat : 8 threads sur 50
    /// positions variées à 500 ms chacune, réseau complet — aucun crash, un
    /// coup légal partout, TT partagée réellement utile (taux de hit
    /// raisonnable, la table est conservée de position en position). Puis
    /// fumée int8 SMP sur 5 positions (piles quantizées PAR THREAD sous
    /// concurrence).
    /// Lancer : cargo test --lib --release smp_8_threads -- --ignored --nocapture
    #[test]
    #[ignore]
    fn smp_8_threads_50_positions_500ms() {
        let limites = Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: 500 };
        let mut r = Recherche::new(Arc::new(Mlp::new(0)), 22);
        r.threads = 8;
        let positions = positions_variees(50);
        let (mut sondes, mut hits, mut noeuds) = (0u64, 0u64, 0u64);
        for (i, pos) in positions.iter().enumerate() {
            let res = r.cherche(pos, limites);
            let coup = res.coup.expect("coup légal attendu");
            assert!(pos.is_legal(&coup), "coup illégal {coup:?} (position {i})");
            assert!(res.score.is_finite(), "score non fini (position {i})");
            sondes += r.tt_sondes;
            hits += r.tt_hits;
            noeuds += res.noeuds;
        }
        let taux = hits as f64 / sondes.max(1) as f64;
        println!(
            "SMP 8 threads : {} positions, {noeuds} nœuds cumulés, TT {hits}/{sondes} hits ({:.1} %)",
            positions.len(),
            100.0 * taux
        );
        assert!(taux >= 0.05, "taux de hit TT anormalement bas : {:.2} %", 100.0 * taux);

        let mut r8 = Recherche::new(Arc::new(Mlp::new(0)), 22);
        r8.threads = 8;
        r8.utilise_int8 = true;
        for pos in positions.iter().take(5) {
            let res = r8.cherche(pos, limites);
            let coup = res.coup.expect("coup légal attendu");
            assert!(pos.is_legal(&coup));
        }
        println!("fumée int8 SMP : 5 positions OK");
    }

    /// Diagnostic (ignoré par défaut) : mise à l'échelle du lazy SMP — nœuds
    /// cumulés et profondeur atteinte à 2 s par position, 1 thread contre 8,
    /// réseau complet. Les nœuds doivent croître nettement ; la profondeur ne
    /// doit jamais reculer de plus d'un pli (aléas d'horloge).
    #[test]
    #[ignore]
    fn diag_smp_scaling_2s() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];
        let limites = Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: 2000 };
        for fen in fens {
            let pos = pos_de_fen(fen);
            let mut ligne = String::new();
            for threads in [1u32, 8] {
                let mut r = Recherche::new(Arc::new(Mlp::new(0)), 24);
                r.threads = threads;
                let res = r.cherche(&pos, limites);
                ligne.push_str(&format!(
                    "  {threads} thread(s) : prof {:>2}, {:>8} nœuds ({:>5.1} % hits TT)",
                    res.profondeur,
                    res.noeuds,
                    100.0 * r.tt_hits as f64 / r.tt_sondes.max(1) as f64
                ));
            }
            println!("{ligne}  {fen}");
        }
    }

    /// (TT géante, ignoré par défaut) Alloue une table de 2^log2 cases
    /// (log2 = ECHEC_TT_LOG2, défaut 28 → 4 Gio à 16 octets/case) et vérifie
    /// qu'elle FONCTIONNE : deux recherches successives de la même position,
    /// la seconde nettement accélérée par les hits. À lancer SEUL, en
    /// release, avec la RAM libre correspondante (2^29 → 8 Gio, 2^30 →
    /// 16 Gio : voir octets_tt).
    #[test]
    #[ignore]
    fn tt_geante_s_alloue_et_fonctionne() {
        let log2: u32 = std::env::var("ECHEC_TT_LOG2")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(28);
        println!(
            "allocation d'une TT de 2^{log2} cases ({:.1} Gio)...",
            octets_tt(log2) as f64 / (1u64 << 30) as f64
        );
        let debut = Instant::now();
        let table = Arc::new(TableTT::new(log2));
        println!("allouée en {:?}", debut.elapsed());
        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let mut r = Recherche::avec_table(reseau_reduit_biaise(), table);
        let a = r.cherche(&pos, limites_prof(5));
        let (s1, h1) = (r.tt_sondes, r.tt_hits);
        let b = r.cherche(&pos, limites_prof(5));
        let (s2, h2) = (r.tt_sondes, r.tt_hits);
        println!(
            "passe 1 : {} nœuds, TT {h1}/{s1} ; passe 2 : {} nœuds, TT {h2}/{s2}",
            a.noeuds, b.noeuds
        );
        assert!(a.coup.is_some() && b.coup.is_some());
        assert!(
            b.noeuds * 2 <= a.noeuds,
            "TT géante inopérante : passe 2 à {} nœuds contre {}",
            b.noeuds,
            a.noeuds
        );
        assert!(h2 > 0, "aucun hit TT sur la seconde passe");
    }

    /// Sonde de performance (ignorée par défaut) : nœuds/s et profondeur en
    /// 100 ms sur la position initiale, réseau complet — à lancer via
    /// `cargo test --lib search:: -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mesure_vitesse() {
        for (nom, net) in [
            ("réseau complet", Arc::new(Mlp::new(0))),
            ("réseau réduit (12 cachés)", reseau_reduit()),
        ] {
            let mut r = Recherche::new(net, 20);
            let debut = Instant::now();
            let res = r.cherche(
                &Chess::default(),
                Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: 100 },
            );
            let d = debut.elapsed();
            println!(
                "[dev] {nom} : profondeur {} , {} noeuds en {:?} ({:.0} noeuds/s)",
                res.profondeur,
                res.noeuds,
                d,
                res.noeuds as f64 / d.as_secs_f64()
            );
        }
    }
}
