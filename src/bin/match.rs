//! Harnais de match : le Champion (réseau maison + recherche SMP) contre le
//! « Fantôme de Deep Blue » (Stockfish bridé par UCI_Elo), en cadence lente,
//! VISIBLE en direct sur le web local.
//!
//! Canal du direct — jumeau minimal du direct self-play (src/direct.rs) :
//! l'entraîneur publie models/live.json ATOMIQUEMENT (.tmp puis rename),
//! serve.exe le sert tel quel sur GET /api/live et web/live.js le lit toutes
//! les 400 ms. Ici, même mécanique avec un fichier dédié :
//! match.exe écrit models/match_live.json à CHAQUE coup, serve.exe le sert
//! sur GET /api/match (page /match, web/match.js). Aucun couplage entre les
//! deux directs : self-play et match cohabitent sans se gêner.
//!
//! Contrat JSON publié (lu par web/match.js) :
//! {"actif": true, "termine": bool, "partie": n (1..), "games": N,
//!  "champion_blanc": bool, "elo_fantome": u32,
//!  "ply": u32, "fen": "...", "last_move": "uci"|null,
//!  "history_san": ["e4", ...],
//!  "history_fen": ["<FEN initiale>", "<FEN après pli 1>", ...]
//!      (history_fen[i] = position après i plis : navigation du viewer),
//!  "pensee": null | {"profondeur": u32, "eval": f32 (CÔTÉ BLANCS, [-1,1]),
//!      "noeuds": u64, "ecoule_ms": u64, "camp": "champion",
//!      "pv": "41...Rxe3 42.fxe3 Rf8"|null}
//!      (réflexion EN COURS du champion : réécrite par le hook d'itération
//!       de la recherche, throttle ~1/s ; remise à null à chaque coup joué ;
//!       "pv" : variante principale de l'itération — marche de TT côté
//!       recherche — en SAN numéroté depuis la position au trait, null si
//!       introuvable),
//!  "ponder": null | {"lances": u64, "justes": u64, "taux": f64,
//!      "ms_cumules": u64}
//!      (null sans --ponder ; sinon compteurs CUMULÉS du match : recherches
//!       de fond démarrées sur le temps de l'adversaire, celles dont la
//!       réponse prédite était le vrai coup joué, taux, et temps de recherche
//!       de fond RÉELLEMENT passé — pas la fenêtre d'attente),
//!  "v_champion": f32|null, "v_fantome": f32|null   (CÔTÉ BLANCS, [-1,1]),
//!  "temps_champion_ms": u64, "temps_fantome_ms": u64  (cumul de la partie),
//!  "movetime_champion_ms": u64, "movetime_fantome_ms": u64,
//!  "score_champion": f32, "score_fantome": f32   (points du match),
//!  "threads": u32, "tt_log2": u32,
//!  "result": null|"1-0"|"0-1"|"1/2-1/2", "result_reason": null|"mat"|...}
//!
//! PGN : une partie par fichier dans --pgn (en-têtes White/Black « Champion »
//! et « Fantôme de Deep Blue (Stockfish UCI_Elo N) », résultat, date, cadence).
//!
//! TT géante : --tt-log2 n alloue 2^n cases de 16 octets (search::octets_tt) —
//! 24 → 256 Mio, 26 → 1 Gio, 28 → 4 Gio, 29 → 8 Gio, 30 → 16 Gio (plafond).
//! Garde : une taille supérieure à la RAM physique DISPONIBLE est refusée
//! avec un message clair (GlobalMemoryStatusEx, sans dépendance).
//!
//! Tables Syzygy : --syzygy <dossier> (ex. engines/syzygy) arme la finale
//! parfaite du champion — racine ≤ 5 pièces jouée par DTZ, sondes WDL dans
//! l'arbre (src/syzygy.rs). Off par défaut : comportement historique strict.
//!
//! Ponder : --ponder (OFF par défaut) fait réfléchir le champion PENDANT le
//! tour du Fantôme. Mécanisme « préchauffage de TT » (src/ponder.rs) : après
//! notre coup C, si la variante principale valait [C, R, ...], une recherche
//! de fond part sur la position après C puis R, sur un chercheur DÉDIÉ posé
//! sur la MÊME table de transposition ; elle est rappelée et JOINTE dès
//! l'arrivée du vrai coup adverse, avant notre recherche. Son score n'est
//! jamais réutilisé : tout le gain vient de la table chaude quand la
//! prédiction est juste.
//!
//! --ponder-threads N : threads de la recherche de FOND. Défaut = la MOITIÉ
//! de --threads-recherche (au moins 1), pas la totalité — voir « partage des
//! cœurs » ci-dessous.
//!
//! Sans --ponder, le MOTEUR est strictement inchangé : `arret_externe` reste
//! None sur tout le chemin de recherche (parité bit à bit, mono-thread comme
//! SMP), pas un thread de fond, pas une entrée de TT de plus. Les SORTIES,
//! elles, gagnent deux marqueurs constants : la balise PGN [Ponder "non"] et
//! le champ "ponder": null du direct (le panneau web masque sa ligne dans ce
//! cas, et absorbe aussi les json d'ancienne génération, sans ce champ). Les
//! artefacts ne sont donc pas octet pour octet ceux d'avant le chantier.
//!
//! PARTAGE DES CŒURS — à savoir avant tout match classé. Le movetime du
//! Fantôme est FIXE : le ponder ne lui vole aucun temps de mur, mais il lui
//! dispute le CPU, le L3 et la bande passante mémoire pendant qu'il réfléchit.
//! Avant ce chantier notre processus était strictement inactif sur son tour.
//! Deux garde-fous, tous deux imparfaits : --ponder-threads borne notre
//! consommation ; et rien ici ne configure le moteur UCI, à qui src/uci.rs
//! n'envoie que UCI_LimitStrength/UCI_Elo — il tourne donc sur 1 thread et
//! 16 Mio de hash par défaut. Si le partage des cœurs doit devenir un choix
//! explicite, c'est un `setoption name Threads/Hash` à poser dans uci.rs, et
//! cela change la force de l'adversaire : à décider, pas à subir.
//!
//! MESURER LE GAIN (A/B) — le ponder est un pari, pas un acquis : sur une
//! prédiction fausse la recherche de fond ÉVINCE des entrées profondes du
//! champion (voir l'en-tête de src/ponder.rs). Protocole : deux séries
//! IDENTIQUES (mêmes --seed, --movetime-*, --uci-elo, --tt-log2,
//! --threads-recherche, --games), l'une avec --ponder, l'autre sans, et
//! comparer le score du match — en lisant au passage le taux publié dans
//! "ponder", qui dit si la prédiction porte.
//!
//! Exemple (fume-test) :
//!   match --games 1 --movetime-champion 2000 --movetime-adversaire 2000 \
//!         --uci-elo 2000 --threads-recherche 4 --tt-log2 24 --int8 \
//!         --syzygy engines/syzygy --ponder

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use shakmaty::fen::Fen;
use shakmaty::san::{San, SanPlus};
use shakmaty::uci::UciMove;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};

use echec::nn::Mlp;
use echec::ponder::{DernierePv, Ponder, StatsPonder};
use echec::search::{octets_tt, InfoIteration, Limites, Recherche};
use echec::uci::UciEngine;

/// Plafond de plis d'une partie (long, exigence du contrat : pas de plafond
/// court) — au-delà, nulle d'arbitrage.
const MAX_PLIES: u32 = 512;
/// Fichier du direct, écrit atomiquement à chaque coup (servi par serve.exe
/// sur GET /api/match — voir l'en-tête du module).
const CHEMIN_LIVE: &str = "models/match_live.json";

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Opt {
    net: String,
    engine: String,
    uci_elo: u32,
    movetime_champion: u64,
    movetime_adversaire: u64,
    threads_recherche: u32,
    tt_log2: u32,
    int8: bool,
    /// Dossier des tables Syzygy 3-4-5 (--syzygy). Vide (défaut) = pas de
    /// tables : comportement historique strict.
    syzygy: String,
    /// Réflexion sur le temps de l'adversaire (--ponder). false (défaut) =
    /// comportement historique strict : le moteur ne pense que sur son tour.
    ponder: bool,
    /// Threads de la recherche de FOND (--ponder-threads). 0 (défaut) = auto :
    /// la MOITIÉ de threads_recherche, au moins 1. Volontairement inférieur au
    /// nombre de threads de la recherche officielle : le movetime du Fantôme
    /// est fixe, donc tout ce que nous consommons sur son tour se prend sur
    /// SES cœurs et SON cache, pas sur son horloge (voir l'en-tête).
    ponder_threads: u32,
    games: usize,
    /// REPRISE D'UNE SÉRIE INTERROMPUE (--partie-depart, --parties-total,
    /// --score-depart) : un match de 6 parties peut être joué en plusieurs
    /// lancements. Sans ces options, comportement historique (série de 1 à
    /// `games`, score parti de 0-0). Elles ne changent QUE la comptabilité
    /// affichée et publiée — numéro de partie, total, score cumulé, numéro de
    /// ronde du PGN, couleur de départ —, jamais le jeu lui-même.
    partie_depart: usize,
    parties_total: usize,
    score_depart_champion: f64,
    score_depart_fantome: f64,
    /// OUVERTURES IMPOSÉES (--livre-plis N) : N premiers demi-coups d'une ligne
    /// du livre joués d'office par les deux camps. 0 (défaut) = position
    /// initiale, comportement historique. Les rondes vont par PAIRES : la même
    /// ouverture est jouée deux fois, une par couleur — l'appariement des
    /// matchs de moteurs, qui annule le biais d'ouverture.
    livre_plis: u32,
    pgn: String,
    /// Réservée (reproductibilité d'une future randomisation : livre,
    /// température...) — la recherche à température 0 est déterministe.
    /// Consignée dans les PGN.
    seed: u64,
}

fn valeur(args: &[String], i: usize, nom: &str) -> String {
    args.get(i + 1).cloned().unwrap_or_else(|| {
        eprintln!("option {nom} : valeur manquante");
        std::process::exit(2);
    })
}

fn parse_valeur<T: std::str::FromStr>(v: &str, nom: &str) -> T {
    v.parse().unwrap_or_else(|_| {
        eprintln!("option {nom} : valeur illisible « {v} »");
        std::process::exit(2);
    })
}

fn parse_args() -> Opt {
    let mut opt = Opt {
        net: "models/chess_best.bin".to_string(),
        engine: "engines/stockfish/stockfish-windows-x86-64-avx2.exe".to_string(),
        uci_elo: 2800,
        movetime_champion: 60_000,
        movetime_adversaire: 60_000,
        threads_recherche: 1,
        tt_log2: 24,
        int8: false,
        syzygy: String::new(),
        ponder: false,
        ponder_threads: 0,
        games: 1,
        partie_depart: 1,
        parties_total: 0,
        score_depart_champion: 0.0,
        score_depart_fantome: 0.0,
        livre_plis: 0,
        pgn: "pgn".to_string(),
        seed: 0xDEE9_B1CE,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        // Drapeaux SANS valeur : n'avancent que d'un cran.
        if nom == "--int8" {
            opt.int8 = true;
            i += 1;
            continue;
        }
        if nom == "--ponder" {
            opt.ponder = true;
            i += 1;
            continue;
        }
        match nom.as_str() {
            "--net" => opt.net = valeur(&args, i, &nom),
            "--engine" => opt.engine = valeur(&args, i, &nom),
            "--uci-elo" => opt.uci_elo = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--movetime-champion" => {
                opt.movetime_champion = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--movetime-adversaire" => {
                opt.movetime_adversaire = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--threads-recherche" => {
                opt.threads_recherche = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--ponder-threads" => {
                opt.ponder_threads = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--tt-log2" => opt.tt_log2 = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--syzygy" => opt.syzygy = valeur(&args, i, &nom),
            "--games" => opt.games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--partie-depart" => {
                opt.partie_depart = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--parties-total" => {
                opt.parties_total = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            // « c,f » : points DÉJÀ acquis par le champion et par le Fantôme.
            "--score-depart" => {
                let v = valeur(&args, i, &nom);
                let (c, f) = v.split_once(',').unwrap_or_else(|| {
                    eprintln!("option --score-depart : attendu « champion,fantome » (ex. 1,1)");
                    std::process::exit(2);
                });
                opt.score_depart_champion = parse_valeur(c.trim(), &nom);
                opt.score_depart_fantome = parse_valeur(f.trim(), &nom);
            }
            "--livre-plis" => opt.livre_plis = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--pgn" => opt.pgn = valeur(&args, i, &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    opt
}

// ---------------------------------------------------------------------------
// Garde mémoire (TT géante)
// ---------------------------------------------------------------------------

/// RAM physique DISPONIBLE en octets (Windows : GlobalMemoryStatusEx, sans
/// dépendance — kernel32 est toujours lié). None si l'appel échoue ou hors
/// Windows : la garde devient un simple avertissement.
#[cfg(windows)]
fn ram_disponible() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(lp_buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut ms = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    // SÛRETÉ : struct #[repr(C)] correctement dimensionnée (dwLength posé),
    // l'API n'écrit que dans ce tampon.
    (unsafe { GlobalMemoryStatusEx(&mut ms) } != 0).then_some(ms.ull_avail_phys)
}

#[cfg(not(windows))]
fn ram_disponible() -> Option<u64> {
    None
}

fn gio(octets: u64) -> f64 {
    octets as f64 / (1u64 << 30) as f64
}

/// Refuse une TT plus grosse que la RAM disponible (message clair) ; simple
/// avertissement si la RAM ne peut pas être mesurée.
fn garde_tt(tt_log2: u32) {
    let octets = octets_tt(tt_log2);
    match ram_disponible() {
        Some(dispo) if octets > dispo => {
            eprintln!(
                "TT refusée : 2^{tt_log2} cases = {:.1} Gio, mais seulement {:.1} Gio de RAM \
                 physique disponible — réduisez --tt-log2 (16 octets/case : 26 → 1 Gio, \
                 28 → 4 Gio, 29 → 8 Gio).",
                gio(octets),
                gio(dispo)
            );
            std::process::exit(2);
        }
        Some(dispo) => println!(
            "TT : 2^{tt_log2} cases = {:.2} Gio ({:.1} Gio de RAM disponible)",
            gio(octets),
            gio(dispo)
        ),
        None => println!(
            "TT : 2^{tt_log2} cases = {:.2} Gio (RAM disponible non mesurable : pas de garde)",
            gio(octets)
        ),
    }
}

// ---------------------------------------------------------------------------
// Direct (jumeau de src/direct.rs, fichier dédié au match)
// ---------------------------------------------------------------------------

/// Écriture atomique du fichier du direct (.tmp puis rename, comme
/// direct.rs) : le lecteur (serve.exe → match.js) ne voit jamais de JSON
/// partiel. Les erreurs d'E/S sont ignorées : le direct ne doit jamais
/// faire tomber le match.
fn ecrit_live(contenu: &str) {
    let tmp = format!("{CHEMIN_LIVE}.tmp");
    if std::fs::write(&tmp, contenu).is_ok() {
        let _ = std::fs::rename(&tmp, CHEMIN_LIVE);
    }
}

/// État permanent du match publié à chaque coup.
struct Direct {
    games: usize,
    champion_blanc: bool,
    partie: usize,
    elo_fantome: u32,
    movetime_champion: u64,
    movetime_fantome: u64,
    threads: u32,
    tt_log2: u32,
    score_champion: f64,
    score_fantome: f64,
    /// Cliché de la DERNIÈRE publication (JSON complet) : le hook « pensée »
    /// de la recherche (thread principal du champion) le relit pour réécrire
    /// le fichier avec le champ "pensee" à jour sans re-connaître tout
    /// l'état de la partie. Mutex par principe (le hook et publie() tournent
    /// en fait sur le même thread) ; Arc pour la capture 'static du hook.
    etat: Arc<Mutex<serde_json::Value>>,
}

impl Direct {
    /// Publie l'état complet dans models/match_live.json (écriture atomique
    /// via ecrit_live) et met le cliché partagé à jour. "pensee" repart à
    /// null : un coup vient d'être joué, la réflexion précédente est close.
    #[allow(clippy::too_many_arguments)]
    fn publie(
        &self,
        termine: bool,
        ply: u32,
        fen: &str,
        last_move: Option<&str>,
        history_san: &[String],
        history_fen: &[String],
        v_champion: Option<f32>,
        v_fantome: Option<f32>,
        temps_champion_ms: u64,
        temps_fantome_ms: u64,
        result: Option<&str>,
        result_reason: Option<&str>,
        ponder: Option<StatsPonder>,
    ) {
        // Bloc "ponder" : null sans --ponder (le panneau web éteint sa ligne).
        let ponder = ponder.map(|s| {
            serde_json::json!({
                "lances": s.lances,
                "justes": s.justes,
                "taux": s.taux(),
                "ms_cumules": s.ms_cumules,
            })
        });
        let v = serde_json::json!({
            "actif": true,
            "termine": termine,
            "partie": self.partie,
            "games": self.games,
            "champion_blanc": self.champion_blanc,
            "elo_fantome": self.elo_fantome,
            "ply": ply,
            "fen": fen,
            "last_move": last_move,
            "history_san": history_san,
            "history_fen": history_fen,
            "pensee": serde_json::Value::Null,
            "ponder": ponder,
            "v_champion": v_champion,
            "v_fantome": v_fantome,
            "temps_champion_ms": temps_champion_ms,
            "temps_fantome_ms": temps_fantome_ms,
            "movetime_champion_ms": self.movetime_champion,
            "movetime_fantome_ms": self.movetime_fantome,
            "score_champion": self.score_champion,
            "score_fantome": self.score_fantome,
            "threads": self.threads,
            "tt_log2": self.tt_log2,
            "result": result,
            "result_reason": result_reason,
        });
        ecrit_live(&v.to_string());
        if let Ok(mut etat) = self.etat.lock() {
            *etat = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Partie
// ---------------------------------------------------------------------------

/// Même convention de hachage que serve.rs/train.rs (règle des 3 répétitions).
fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Score d'une recherche (point de vue du TRAIT, mats à ±SCORE_MAT) → jauge
/// CÔTÉ BLANCS dans [-1, 1] pour le direct.
fn vers_blancs(score_trait: f32, trait_blanc: bool) -> f32 {
    let v = if trait_blanc { score_trait } else { -score_trait };
    v.clamp(-1.0, 1.0)
}

/// Ligne SAN NUMÉROTÉE d'une variante principale depuis `pos` (la position
/// au trait, racine de la réflexion) : « 41...Rxe3 42.fxe3 Rf8 » — numéro de
/// coup devant chaque coup blanc, « N... » en tête si le trait est aux noirs.
/// Réutilise SanPlus (le SAN du PGN, suffixes +/# compris). Coup illégal
/// (TT bruitée, FEN désynchronisée) → troncature propre sur ce qui précède.
fn pv_en_san(pos: &Chess, pv: &[Move]) -> String {
    let mut p = pos.clone();
    let mut ligne = String::new();
    for (i, m) in pv.iter().enumerate() {
        if !p.is_legal(m) {
            break; // ceinture : on n'affiche jamais une suite illégale
        }
        let num = p.fullmoves();
        if p.turn() == Color::White {
            if i > 0 {
                ligne.push(' ');
            }
            ligne.push_str(&format!("{num}."));
        } else if i == 0 {
            ligne.push_str(&format!("{num}..."));
        } else {
            ligne.push(' ');
        }
        let san = SanPlus::from_move(p.clone(), m).to_string();
        ligne.push_str(&san);
        p.play_unchecked(m);
    }
    ligne
}

/// Issue d'une partie (résultat + dernier état publié, pour la publication
/// finale « match terminé » qui garde la position réelle à l'écran).
struct Issue {
    result: &'static str,
    reason: &'static str,
    history_san: Vec<String>,
    /// FEN après chaque pli, position initiale incluse (history_fen[i] =
    /// après i plis) : navigation du viewer, republiée à la fin du match.
    history_fen: Vec<String>,
    /// Points du champion : 1.0 / 0.5 / 0.0.
    points_champion: f64,
    fen: String,
    last_move: Option<String>,
    v_champion: Option<f32>,
    v_fantome: Option<f32>,
    temps_champion_ms: u64,
    temps_fantome_ms: u64,
}

/// Vérifie l'issue de la position APRÈS un coup (compte de répétitions déjà
/// mis à jour). None : la partie continue.
fn arbitre(pos: &Chess, repetitions: u32, plies: u32) -> Option<(&'static str, &'static str)> {
    if pos.is_checkmate() {
        // Le camp au trait est maté : celui qui vient de jouer gagne.
        return Some(match pos.turn() {
            Color::White => ("0-1", "mat"),
            Color::Black => ("1-0", "mat"),
        });
    }
    if pos.is_stalemate() {
        return Some(("1/2-1/2", "pat"));
    }
    if pos.is_insufficient_material() {
        return Some(("1/2-1/2", "matériel insuffisant"));
    }
    if pos.halfmoves() >= 100 {
        return Some(("1/2-1/2", "50 coups"));
    }
    if repetitions >= 3 {
        return Some(("1/2-1/2", "3 répétitions"));
    }
    if plies >= MAX_PLIES {
        return Some(("1/2-1/2", "plafond de 512 plis"));
    }
    None
}

/// Joue UNE partie complète, publie le direct à chaque coup, renvoie l'issue.
/// Le champion garde sa TT entre les coups (nouvelle_partie() la vide entre
/// deux parties) ; le Fantôme reçoit un ucinewgame par partie.
///
/// PONDER (src/ponder.rs, `ponder` éteint = comportement historique strict) :
/// après CHAQUE coup du champion, si la variante principale de la recherche
/// prédit une réponse, une recherche de fond part sur la position
/// correspondante et occupe le temps de réflexion du Fantôme ; elle est
/// rappelée et JOINTE dès l'arrivée du coup adverse, donc AVANT la recherche
/// suivante du champion. Invariants tenus ici :
///   - au plus une recherche de fond en vol, jamais pendant notre recherche
///     ni entre deux parties (la TT y est vidée) ;
///   - toutes les sorties de la boucle passent par un arrêt du ponder ;
///   - le chercheur de fond n'a pas de hook `info` : il n'écrit JAMAIS dans
///     models/match_live.json, dont la boucle reste seule propriétaire.
fn partie(
    recherche: &mut Recherche,
    moteur: &mut UciEngine,
    direct: &Direct,
    movetime_champion: u64,
    movetime_fantome: u64,
    ponder: &mut Ponder,
    derniere_pv: &DernierePv,
    ouverture: &[String],
) -> Issue {
    let champion_blanc = direct.champion_blanc;
    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u32> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    let mut history_san: Vec<String> = Vec::new();
    let mut history_fen: Vec<String> = Vec::new();
    let mut plies: u32 = 0;
    let mut temps_champion_ms: u64 = 0;
    let mut temps_fantome_ms: u64 = 0;
    let mut v_champion: Option<f32> = None;
    let mut v_fantome: Option<f32> = None;
    let mut dernier_uci: Option<String> = None;

    // Position initiale au micro dès le premier tick de la page (elle ouvre
    // aussi history_fen : la navigation remonte jusqu'à elle).
    let fen0 = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
    history_fen.push(fen0.clone());

    // OUVERTURE IMPOSÉE (livre) : les deux camps la jouent d'office, aucun
    // moteur n'est consulté. Sans elle, deux moteurs déterministes rejouent la
    // même partie à chaque fois — seuls les aléas de minutage les font diverger.
    // Les coups entrent dans l'historique comme les autres : le PGN reste une
    // partie complète depuis le premier coup, et les répétitions comptent bien
    // les positions traversées.
    for san_txt in ouverture {
        let Ok(san) = San::from_ascii(san_txt.as_bytes()) else {
            panic!("coup d'ouverture illisible : {san_txt}");
        };
        let Ok(coup) = san.to_move(&pos) else {
            panic!("coup d'ouverture illégal : {san_txt}");
        };
        let san_complet = SanPlus::from_move(pos.clone(), &coup).to_string();
        pos.play_unchecked(&coup);
        history_san.push(san_complet);
        plies += 1;
        *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
        history_fen.push(Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string());
    }
    let fen0 = history_fen.last().cloned().unwrap_or(fen0);

    direct.publie(
        false,
        plies,
        &fen0,
        None,
        &history_san,
        &history_fen,
        None,
        None,
        0,
        0,
        None,
        None,
        ponder.stats_publiables(),
    );

    let limites_champion = Limites {
        max_noeuds: 0,
        max_profondeur: 0,
        movetime_ms: movetime_champion,
    };

    let (result, reason) = loop {
        let trait_blanc = pos.turn() == Color::White;
        let tour_champion = trait_blanc == champion_blanc;
        let debut = Instant::now();
        // Réponse prédite par la recherche du champion (ponder) : posée à
        // notre tour, consommée après que le coup a été joué.
        let mut prediction: Option<Move> = None;

        let coup: Move = if tour_champion {
            // Ceinture : aucune recherche de fond ne doit tourner pendant la
            // nôtre (elle est normalement déjà jointe — arrivée du coup
            // adverse — ce rappel est un no-op).
            ponder.arrete(None);
            // La boîte aux lettres repart vide : jamais de prédiction bâtie
            // sur la variante principale du coup précédent.
            derniere_pv.vide();
            let res = recherche.cherche(&pos, limites_champion);
            v_champion = Some(vers_blancs(res.score, trait_blanc));
            temps_champion_ms += debut.elapsed().as_millis() as u64;
            let coup = res.coup.expect("coups légaux non vides (l'arbitre a déjà statué)");
            prediction = derniere_pv.reponse_apres(&coup);
            coup
        } else {
            let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
            let reponse = moteur.meilleur_coup_et_score_fen(&fen, movetime_fantome);
            let ecoule_fantome = debut.elapsed().as_millis() as u64;
            // Parse UCI + validation de LÉGALITÉ contre la position : un coup
            // illégal du moteur vaut forfait, jamais de corruption
            // silencieuse de la partie.
            let coup_adverse = match &reponse {
                Ok((texte, _)) => UciMove::from_ascii(texte.as_bytes())
                    .ok()
                    .and_then(|u| u.to_move(&pos).ok())
                    .filter(|m| pos.legal_moves().contains(m)),
                Err(_) => None,
            };
            // LE COUP ADVERSE EST LÀ : la recherche de fond est rappelée et
            // JOINTE ici, avant tout le reste — y compris avant les sorties
            // par forfait ci-dessous. Le coup sert à créditer la prédiction.
            ponder.arrete(coup_adverse.as_ref());
            match reponse {
                Ok((texte, score)) => {
                    v_fantome = score.map(|s| vers_blancs(s, trait_blanc));
                    temps_fantome_ms += ecoule_fantome;
                    match coup_adverse {
                        Some(m) => m,
                        None => {
                            // Le trait est au Fantôme : le forfait donne la
                            // partie au champion, quelle que soit sa couleur.
                            eprintln!("coup illégal du Fantôme ({texte:?} sur {fen}) : forfait");
                            break if trait_blanc {
                                ("0-1", "forfait (coup illégal du moteur)")
                            } else {
                                ("1-0", "forfait (coup illégal du moteur)")
                            };
                        }
                    }
                }
                Err(e) => {
                    eprintln!("moteur UCI en erreur ({e}) : forfait du Fantôme");
                    break if trait_blanc {
                        ("0-1", "forfait (moteur en erreur)")
                    } else {
                        ("1-0", "forfait (moteur en erreur)")
                    };
                }
            }
        };

        // SAN (suffixes +/# compris) calculé AVANT de jouer.
        let san = SanPlus::from_move(pos.clone(), &coup).to_string();
        let uci = coup.to_uci(CastlingMode::Standard).to_string();
        pos.play_unchecked(&coup);
        history_san.push(san);
        plies += 1;
        let compte = {
            let entree = repetitions.entry(zobrist(&pos)).or_insert(0);
            *entree += 1;
            *entree
        };

        dernier_uci = Some(uci.clone());
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        history_fen.push(fen.clone());
        let verdict = arbitre(&pos, compte, plies);
        direct.publie(
            false,
            plies,
            &fen,
            Some(&uci),
            &history_san,
            &history_fen,
            v_champion,
            v_fantome,
            temps_champion_ms,
            temps_fantome_ms,
            verdict.map(|(r, _)| r),
            verdict.map(|(_, raison)| raison),
            ponder.stats_publiables(),
        );
        if let Some((r, raison)) = verdict {
            break (r, raison);
        }
        // PONDER : notre coup vient d'être joué, la partie continue, et c'est
        // maintenant au Fantôme de consommer son temps de réflexion. Si la
        // variante principale prédit une réponse, la recherche de fond part
        // sur la position qui suivrait — la légalité de la réponse est
        // revérifiée par `demarre` contre `pos`.
        if let Some(reponse) = prediction {
            ponder.demarre(&pos, &reponse);
        }
    };
    // Toute sortie de boucle est déjà passée par un arrêt du ponder ; ce
    // rappel final ferme le cas des sorties par forfait ET garantit qu'aucun
    // thread de fond ne franchit la fin de partie (la TT est vidée juste
    // après, par nouvelle_partie()).
    ponder.arrete(None);

    let points_blancs = match result {
        "1-0" => 1.0,
        "0-1" => 0.0,
        _ => 0.5,
    };
    Issue {
        result,
        reason,
        history_san,
        history_fen,
        points_champion: if champion_blanc { points_blancs } else { 1.0 - points_blancs },
        fen: Fen::from_position(pos, EnPassantMode::Legal).to_string(),
        last_move: dernier_uci,
        v_champion,
        v_fantome,
        temps_champion_ms,
        temps_fantome_ms,
    }
}

// ---------------------------------------------------------------------------
// PGN
// ---------------------------------------------------------------------------

/// Écrit le PGN complet d'une partie (en-têtes du contrat + coups SAN pliés
/// à ~80 colonnes, résultat en queue).
#[allow(clippy::too_many_arguments)]
fn ecrit_pgn(
    chemin: &str,
    round: usize,
    champion_blanc: bool,
    elo: u32,
    issue: &Issue,
    opt: &Opt,
    date: &str,
) -> std::io::Result<()> {
    let champion = "Champion";
    let fantome = format!("Fantôme de Deep Blue (Stockfish UCI_Elo {elo})");
    let (blanc, noir) = if champion_blanc {
        (champion.to_string(), fantome)
    } else {
        (fantome, champion.to_string())
    };
    let mut f = std::fs::File::create(chemin)?;
    writeln!(f, "[Event \"Match Champion vs Fantôme de Deep Blue\"]")?;
    writeln!(f, "[Site \"local (match.exe)\"]")?;
    writeln!(f, "[Date \"{date}\"]")?;
    writeln!(f, "[Round \"{round}\"]")?;
    writeln!(f, "[White \"{blanc}\"]")?;
    writeln!(f, "[Black \"{noir}\"]")?;
    writeln!(f, "[Result \"{}\"]", issue.result)?;
    // Cadence : movetime fixe par coup (pas un TimeControl PGN standard,
    // consigné en clair + étiquettes dédiées).
    writeln!(f, "[TimeControl \"-\"]")?;
    writeln!(f, "[MovetimeChampionMs \"{}\"]", opt.movetime_champion)?;
    writeln!(f, "[MovetimeAdversaireMs \"{}\"]", opt.movetime_adversaire)?;
    writeln!(f, "[ThreadsRecherche \"{}\"]", opt.threads_recherche)?;
    writeln!(f, "[TTLog2 \"{}\"]", opt.tt_log2)?;
    writeln!(
        f,
        "[Syzygy \"{}\"]",
        if opt.syzygy.is_empty() { "-" } else { opt.syzygy.as_str() }
    )?;
    writeln!(f, "[Ponder \"{}\"]", if opt.ponder { "oui" } else { "non" })?;
    writeln!(f, "[Seed \"{}\"]", opt.seed)?;
    writeln!(f, "[Termination \"{}\"]", issue.reason)?;
    writeln!(f)?;
    let mut ligne = String::new();
    for (i, san) in issue.history_san.iter().enumerate() {
        let jeton = if i % 2 == 0 {
            format!("{}. {san}", i / 2 + 1)
        } else {
            san.clone()
        };
        if ligne.len() + jeton.len() + 1 > 80 {
            writeln!(f, "{ligne}")?;
            ligne.clear();
        }
        if !ligne.is_empty() {
            ligne.push(' ');
        }
        ligne.push_str(&jeton);
    }
    if !ligne.is_empty() {
        ligne.push(' ');
    }
    ligne.push_str(issue.result);
    writeln!(f, "{ligne}")?;
    Ok(())
}

/// Date du jour au format PGN (AAAA.MM.JJ), sans dépendance : dérivée du
/// temps UNIX (calendrier grégorien proleptique, suffisant pour un en-tête).
fn date_pgn() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let jours = secs / 86_400;
    // Algorithme civil de Howard Hinnant (days_from_civil inversé).
    let z = jours as i64 + 719_468;
    let ere = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + ere * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let j = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let a = if m <= 2 { y + 1 } else { y };
    format!("{a:04}.{m:02}.{j:02}")
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    echec::pleine_puissance(); // la réflexion du match jamais bridée par l'EcoQoS
    let opt = parse_args();

    // Garde TT AVANT toute allocation (message clair plutôt qu'un OOM).
    garde_tt(opt.tt_log2);

    let net = Arc::new(
        Mlp::load(&opt.net)
            .unwrap_or_else(|e| panic!("échec de chargement du réseau {} : {e}", opt.net)),
    );
    println!(
        "champion : {} (architecture {:?}, schéma {:?}), {} threads, movetime {} ms{}",
        opt.net,
        net.sizes,
        net.schema(),
        opt.threads_recherche,
        opt.movetime_champion,
        if opt.int8 { ", int8" } else { "" }
    );

    // UN chercheur pour tout le match : la TT géante est allouée une fois,
    // vidée entre les parties (nouvelle_partie), réutilisée entre les coups.
    let mut recherche = Recherche::new(net, opt.tt_log2);
    recherche.threads = opt.threads_recherche;
    recherche.utilise_int8 = opt.int8;

    // Tables Syzygy (--syzygy <dossier>, off par défaut) : chargement UNE
    // FOIS ; une erreur est une erreur de configuration — sortie propre
    // plutôt qu'un match sans l'assurance-finales demandée.
    if !opt.syzygy.is_empty() {
        match echec::syzygy::charge(&opt.syzygy) {
            Ok((tables, n)) => {
                println!("syzygy : {n} tables chargées depuis {}", opt.syzygy);
                recherche.syzygy = Some(Arc::new(tables));
            }
            Err(e) => {
                eprintln!("--syzygy {} : {e}", opt.syzygy);
                std::process::exit(2);
            }
        }
    }

    // PONDER (--ponder, off par défaut) : un chercheur DÉDIÉ, construit
    // APRÈS le réglage des drapeaux du champion (threads, int8, syzygy) qu'il
    // recopie, et surtout posé sur la MÊME TABLE DE TRANSPOSITION (Arc
    // partagé) — c'est ce partage, et lui seul, qui transporte le travail de
    // la réflexion de fond vers la recherche officielle du coup suivant.
    // Sans l'option : Ponder::eteint(), pas un thread, pas un octet de plus.
    //
    // THREADS DU FOND : volontairement RÉDUITS par rapport à la recherche
    // officielle (défaut : la moitié). Le movetime du Fantôme est fixe — ce
    // que nous prenons sur son tour ne se voit sur aucune horloge, seulement
    // sur ses nœuds, c'est-à-dire sur sa force. Laisser des cœurs libres est
    // le seul garde-fou que ce harnais possède (voir l'en-tête).
    let mut ponder = if opt.ponder {
        let n = if opt.ponder_threads > 0 {
            opt.ponder_threads
        } else {
            (opt.threads_recherche / 2).max(1)
        };
        if n >= opt.threads_recherche && opt.threads_recherche > 1 {
            println!(
                "ponder : ATTENTION — {n} thread(s) de fond pour {} de recherche : le Fantôme \
                 réfléchira sur une machine saturée, sa force en pâtira sans que son horloge \
                 le montre",
                opt.threads_recherche
            );
        }
        println!(
            "ponder : actif (réflexion sur le temps du Fantôme, TT partagée, {n} thread(s) \
             de fond contre {} de recherche)",
            opt.threads_recherche
        );
        let mut fond = recherche.jumeau_meme_tt();
        fond.threads = n;
        Ponder::arme(fond)
    } else {
        Ponder::eteint()
    };

    let mut moteur = UciEngine::lance(&opt.engine)
        .unwrap_or_else(|e| panic!("échec de lancement du moteur {} : {e}", opt.engine));
    let elo = moteur
        .limite_force(opt.uci_elo)
        .unwrap_or_else(|e| panic!("échec de réglage UCI_Elo : {e}"));
    if elo != opt.uci_elo {
        println!("UCI_Elo demandé {} clampé aux bornes du moteur : {elo}", opt.uci_elo);
    }
    println!(
        "fantôme : {} (UCI_Elo {elo}), movetime {} ms",
        opt.engine, opt.movetime_adversaire
    );

    std::fs::create_dir_all(&opt.pgn)
        .unwrap_or_else(|e| panic!("impossible de créer le dossier PGN {} : {e}", opt.pgn));
    let date = date_pgn();

    // Comptabilité de la série : `parties_total` par défaut = la longueur de
    // CE lancement (comportement historique) ; sinon le total du match complet.
    let parties_total = if opt.parties_total == 0 {
        opt.partie_depart + opt.games - 1
    } else {
        opt.parties_total
    };
    let mut direct = Direct {
        games: parties_total,
        champion_blanc: true,
        partie: 0,
        elo_fantome: elo,
        movetime_champion: opt.movetime_champion,
        movetime_fantome: opt.movetime_adversaire,
        threads: opt.threads_recherche,
        tt_log2: opt.tt_log2,
        score_champion: opt.score_depart_champion,
        score_fantome: opt.score_depart_fantome,
        etat: Arc::new(Mutex::new(serde_json::Value::Null)),
    };

    // Hook « pensée en direct » (Recherche::info) : à chaque itération
    // d'approfondissement TERMINÉE du champion, la dernière publication est
    // réécrite avec un objet "pensee" à jour (profondeur, éval CÔTÉ BLANCS,
    // nœuds du thread principal, écoulé). Écriture atomique (ecrit_live),
    // throttle ~1 écriture/s : à 3 min/coup, largement assez vivant sans
    // marteler le disque. Le hook est hors chemin chaud (entre itérations)
    // et n'existe que sur la recherche du champion — jamais sur le Fantôme.
    //
    // Le hook sert AUSSI de source au ponder : la variante principale de
    // chaque itération est déposée dans `derniere_pv` (boîte aux lettres de
    // src/ponder.rs) AVANT le throttle — la dernière PV déposée est donc
    // toujours celle de l'itération qui a fourni le coup joué, quel que soit
    // le rythme d'écriture du direct. Coût : une copie de ≤ 12 coups par
    // itération, hors chemin chaud.
    let etat_live = direct.etat.clone();
    let derniere_pv = DernierePv::nouvelle();
    let pv_pour_hook = derniere_pv.clone();
    let derniere_ecriture: Mutex<Option<Instant>> = Mutex::new(None);
    recherche.info = Some(Box::new(move |it: InfoIteration| {
        pv_pour_hook.depose(&it.pv);
        if let Ok(mut t) = derniere_ecriture.lock() {
            if t.is_some_and(|t0| t0.elapsed().as_millis() < 1000) {
                return; // throttle : au plus ~1 écriture par seconde
            }
            *t = Some(Instant::now());
        }
        let Ok(mut v) = etat_live.lock() else { return };
        if v.is_null() {
            return; // rien encore publié : pas d'état à enrichir
        }
        // Le penseur est TOUJOURS le champion (le hook n'est posé que sur sa
        // recherche) : score perspective du trait → côté blancs via la
        // couleur du champion dans la partie en cours, clampé comme les
        // jauges (les mats saturent à ±1).
        let champion_blanc = v["champion_blanc"].as_bool().unwrap_or(true);
        let eval = (if champion_blanc { it.score } else { -it.score }).clamp(-1.0, 1.0);
        // PV en SAN numéroté : la racine de la réflexion est la position de
        // la dernière publication (v["fen"] — publie() précède toujours le
        // cherche() du champion). FEN illisible ou PV vide → null, le
        // panneau web éteint simplement la ligne.
        let pv_san = v["fen"]
            .as_str()
            .and_then(|f| f.parse::<Fen>().ok())
            .and_then(|f| f.into_position::<Chess>(CastlingMode::Standard).ok())
            .map(|pos| pv_en_san(&pos, &it.pv))
            .filter(|l| !l.is_empty());
        v["pensee"] = serde_json::json!({
            "profondeur": it.profondeur,
            "eval": eval,
            "noeuds": it.noeuds,
            "ecoule_ms": it.ecoule_ms,
            "camp": "champion",
            "pv": pv_san,
        });
        ecrit_live(&v.to_string());
    }));

    for i in 0..opt.games {
        // Numéro de la partie DANS LE MATCH (pas dans ce lancement) : c'est lui
        // qui gouverne l'affichage, le PGN et la couleur, pour qu'une série
        // reprise garde l'alternance du match complet.
        let numero = opt.partie_depart + i;
        // Couleurs alternées : le champion a les blancs aux parties impaires
        // (1re, 3e, ...).
        direct.partie = numero;
        direct.champion_blanc = numero % 2 == 1;
        // AVANT le vidage de la table : joint un éventuel fond en vol (partie()
        // le fait déjà — ceinture) ET remet à zéro les killers/historique du
        // chercheur de FOND, que `recherche.nouvelle_partie()` ne touche pas
        // (ils sont par chercheur, pas dans la table partagée). Sans ce rappel,
        // le fond traverserait tout le match avec des heuristiques héritées de
        // parties révolues.
        ponder.nouvelle_partie();
        recherche.nouvelle_partie();
        if let Err(e) = moteur.nouvelle_partie() {
            panic!("ucinewgame en erreur avant la partie {numero} : {e}");
        }

        println!(
            "— partie {}/{} : champion avec les {}",
            numero,
            parties_total,
            if direct.champion_blanc { "blancs" } else { "noirs" }
        );
        // Ouverture de la ronde : une par PAIRE de parties (3 et 4 partagent
        // la même, 5 et 6 la suivante...), tirée du livre par la graine du
        // match. Les deux couleurs jouent donc la même position de départ :
        // tout biais de l'ouverture se compense entre les deux rondes.
        let ouverture: Vec<String> = if opt.livre_plis == 0 {
            Vec::new()
        } else {
            let lignes = echec::departs::lignes_du_livre();
            let paire = (numero - 1) / 2;
            let idx = (opt.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                >> 17)
                .wrapping_add(paire as u64) as usize
                % lignes.len();
            let (etiquette, coups) = lignes[idx];
            println!("  ouverture imposée : {etiquette}");
            coups
                .split_whitespace()
                .take(opt.livre_plis as usize)
                .map(|s| s.to_string())
                .collect()
        };
        let issue = partie(
            &mut recherche,
            &mut moteur,
            &direct,
            opt.movetime_champion,
            opt.movetime_adversaire,
            &mut ponder,
            &derniere_pv,
            &ouverture,
        );
        debug_assert!(ponder.au_repos(), "thread de ponder survivant à la partie");
        direct.score_champion += issue.points_champion;
        direct.score_fantome += 1.0 - issue.points_champion;

        let chemin_pgn = format!("{}/partie_{:03}.pgn", opt.pgn, numero);
        ecrit_pgn(&chemin_pgn, numero, direct.champion_blanc, elo, &issue, &opt, &date)
            .unwrap_or_else(|e| eprintln!("écriture PGN {chemin_pgn} en échec : {e}"));
        println!(
            "  {} ({}) en {} plis — PGN : {chemin_pgn} — score Champion {} · Fantôme {}",
            issue.result,
            issue.reason,
            issue.history_san.len(),
            direct.score_champion,
            direct.score_fantome
        );

        // Après la DERNIÈRE partie : re-publication de l'état final marquée
        // « match terminé » (le score inclut désormais cette partie, la page
        // affiche la position réelle et le score final).
        if i + 1 == opt.games {
            direct.publie(
                true,
                issue.history_san.len() as u32,
                &issue.fen,
                issue.last_move.as_deref(),
                &issue.history_san,
                &issue.history_fen,
                issue.v_champion,
                issue.v_fantome,
                issue.temps_champion_ms,
                issue.temps_fantome_ms,
                Some(issue.result),
                Some(issue.reason),
                ponder.stats_publiables(),
            );
        }
    }

    println!(
        "match terminé : Champion {} — Fantôme {} ({} partie(s))",
        direct.score_champion, direct.score_fantome, opt.games
    );
    if let Some(s) = ponder.stats_publiables() {
        println!(
            "ponder : {} lancé(s), {} juste(s) ({:.1} %), {:.1} s de recherche de fond \
             effective sur le temps du Fantôme",
            s.lances,
            s.justes,
            100.0 * s.taux(),
            s.ms_cumules as f64 / 1000.0
        );
    }
}
