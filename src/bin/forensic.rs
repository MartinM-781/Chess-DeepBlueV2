//! Harnais forensique NNUE : le MÊME réseau s'affronte lui-même à NOMBRE DE
//! NŒUDS FIXE — camp A évalué par les accumulateurs incrémentaux
//! (`utilise_nnue = true`, chemin normal), camp B par le forward complet
//! recalculé à chaque feuille (`utilise_nnue = false`, chemin de secours).
//! À budget de nœuds identique la vitesse ne compte pas : si l'incrémental
//! rend les MÊMES évaluations, les deux camps jouent la même force et le
//! score converge vers 50 % ; tout écart significatif prouve des évaluations
//! FAUSSES quelque part dans src/nnue.rs.
//!
//! Schéma de duel copié du gating de train.rs : chaque PAIRE de parties tire
//! une ouverture aléatoire de quelques plis, jouée une fois de chaque couleur
//! (équité), bots déterministes à température 0, mêmes règles d'arbitrage
//! (mat/pat, matériel insuffisant, 50 coups, 3e répétition, plafond de plis).
//!
//! Usage :
//!   forensic --net models/chess_latest.bin --games 200 --nodes 8000 \
//!            --threads 4 --out forensic.csv [--seed 42]
//!
//! Sortie : lignes CSV `net,partie,couleur_nnue,resultat` (résultat du point
//! de vue du camp NNUE : 1, 0.5, 0) en APPEND dans --out, et récapitulatif
//! final sur stdout.

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Move, Position};

use echec::nn::Mlp;
use echec::search::{Limites, Recherche};

/// Plis max d'une partie (ouverture comprise), comme le gating de train.rs.
const MAX_PLIES: u32 = 300;
/// Plis d'ouverture aléatoires partagés par chaque paire (même valeur que
/// PLIS_OUVERTURE_GATING de train.rs : diversification des trajectoires,
/// les deux camps étant déterministes à température 0).
const PLIS_OUVERTURE: u32 = 4;
/// Taille de la TT de chaque chercheur : 2^20 ≈ 1M d'entrées (comme
/// BotRecherche).
const TAILLE_TT_LOG2: u32 = 20;

/// Même convention de hachage que train.rs/arena (règle des 3 répétitions).
fn zobrist(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

/// Même mélange de graine que train.rs (splitmix-like).
fn derive_graine(base: u64, sel: u64) -> u64 {
    let mut x = base ^ sel.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    x
}

struct Opt {
    net: String,
    games: usize,
    nodes: u64,
    threads: usize,
    out: String,
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
        net: "models/chess_latest.bin".to_string(),
        games: 200,
        nodes: 8000,
        threads: 4,
        out: "forensic.csv".to_string(),
        seed: 0xF0E5_1C42,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--net" => opt.net = valeur(&args, i, &nom),
            "--games" => opt.games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--nodes" => opt.nodes = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--threads" => opt.threads = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--out" => opt.out = valeur(&args, i, &nom),
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

/// Fabrique un chercheur d'un camp : même réseau, seule la voie d'évaluation
/// change. C'est TOUTE la différence entre les deux camps.
fn chercheur(net: &Arc<Mlp>, incremental: bool) -> Recherche {
    let mut r = Recherche::new(net.clone(), TAILLE_TT_LOG2);
    r.utilise_nnue = incremental;
    r
}

/// Une partie : camp NNUE (incrémental) contre camp exact (forward complet),
/// même réseau, `nodes` nœuds par coup, température 0 (coup du chercheur tel
/// quel, déterministe). Chercheurs FRAIS (TT vierges, équité). Arbitrage
/// identique à partie_gating de train.rs. Renvoie les points du camp NNUE
/// (1.0 victoire, 0.5 nulle, 0.0 défaite).
fn partie(
    net: &Arc<Mlp>,
    nnue_blanc: bool,
    ouverture: &[Move],
    nodes: u64,
) -> f32 {
    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    for m in ouverture {
        pos = pos.play(m).expect("coup d'ouverture légal");
        *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
    }
    let limites = Limites { max_noeuds: nodes, max_profondeur: 0, movetime_ms: 0 };
    let mut camp_nnue = chercheur(net, true);
    let mut camp_exact = chercheur(net, false);
    let mut plies = ouverture.len() as u32;

    let resultat_blancs = loop {
        let coups = pos.legal_moves();
        if coups.is_empty() {
            // Mat : le trait est perdant ; pat : nulle.
            break if pos.is_check() {
                if pos.turn() == Color::White { -1.0 } else { 1.0 }
            } else {
                0.0
            };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= MAX_PLIES {
            break 0.0;
        }
        let tour_nnue = (pos.turn() == Color::White) == nnue_blanc;
        let res = if tour_nnue {
            camp_nnue.cherche(&pos, limites)
        } else {
            camp_exact.cherche(&pos, limites)
        };
        let m = res.coup.expect("coups légaux non vides");
        pos = pos.play(&m).expect("coup légal");
        plies += 1;
        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            break 0.0;
        }
    };
    let cote = if nnue_blanc { resultat_blancs } else { -resultat_blancs };
    (cote + 1.0) / 2.0
}

fn main() {
    let opt = parse_args();
    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .expect("pool rayon global");

    let net = Arc::new(Mlp::load(&opt.net).unwrap_or_else(|e| {
        eprintln!("--net {} : chargement impossible ({e})", opt.net);
        std::process::exit(1);
    }));
    println!(
        "forensic : net={} sizes={:?} games={} nodes={} threads={} out={} seed={}",
        opt.net, net.sizes, opt.games, opt.nodes, opt.threads, opt.out, opt.seed
    );

    let paires = opt.games / 2;
    if paires == 0 {
        eprintln!("--games {} : il faut au moins une paire (2 parties)", opt.games);
        std::process::exit(2);
    }

    // Paires en parallèle, une par tâche rayon (même remarque que le gating :
    // sans with_max_len(1), les paquets séquentiels sous-occupent le pool).
    let faites = AtomicUsize::new(0);
    // (indice de partie, couleur du camp NNUE, points NNUE) pour le CSV.
    let resultats: Vec<(usize, &'static str, f32)> = (0..paires)
        .into_par_iter()
        .with_max_len(1)
        .flat_map(|p| {
            // Ouverture aléatoire partagée par la paire, jouée des deux
            // couleurs. Jamais à court de coups en 4 plis.
            let mut rng = StdRng::seed_from_u64(derive_graine(opt.seed, p as u64));
            let mut pos = Chess::default();
            let mut ouverture: Vec<Move> = Vec::new();
            for _ in 0..PLIS_OUVERTURE {
                let Some(m) = pos.legal_moves().choose(&mut rng).cloned() else {
                    break;
                };
                pos = pos.play(&m).expect("coup légal");
                ouverture.push(m);
            }
            let pts_blanc = partie(&net, true, &ouverture, opt.nodes);
            let pts_noir = partie(&net, false, &ouverture, opt.nodes);
            let n = faites.fetch_add(1, Ordering::Relaxed) + 1;
            if n % 4 == 0 || n == paires {
                println!("  forensic : {n}/{paires} paires jouees");
                std::io::stdout().flush().ok();
            }
            vec![(2 * p, "blanc", pts_blanc), (2 * p + 1, "noir", pts_noir)]
        })
        .collect();

    // Append CSV (en-tête seulement si le fichier n'existe pas encore).
    let entete = !std::path::Path::new(&opt.out).exists();
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&opt.out)
        .unwrap_or_else(|e| {
            eprintln!("--out {} : ouverture impossible ({e})", opt.out);
            std::process::exit(1);
        });
    if entete {
        writeln!(f, "net,partie,couleur_nnue,resultat").expect("écriture CSV");
    }
    for (idx, couleur, pts) in &resultats {
        writeln!(f, "{},{},{},{}", opt.net, idx, couleur, pts).expect("écriture CSV");
    }

    // Récapitulatif : points du camp NNUE (incrémental) vs camp exact.
    let total = resultats.len() as f32;
    let pts_nnue: f32 = resultats.iter().map(|(_, _, p)| p).sum();
    let (mut v, mut n, mut d) = (0u32, 0u32, 0u32);
    for (_, _, p) in &resultats {
        if *p > 0.75 {
            v += 1;
        } else if *p < 0.25 {
            d += 1;
        } else {
            n += 1;
        }
    }
    println!(
        "forensic : NNUE {pts_nnue:.1} / {total:.0} ({:.1} %) — V {v} / N {n} / D {d} \
         (exact : {:.1} pts, {:.1} %)",
        100.0 * pts_nnue / total,
        total - pts_nnue,
        100.0 * (total - pts_nnue) / total,
    );
    println!(
        "interpretation : ~50 % attendu si l'incremental est fidele ; un ecart \
         significatif a nombre de noeuds fixe prouve des evaluations faussees."
    );
}
