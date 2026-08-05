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
//!
//! Mode duel A/B (`--net-b <chemin>`) : au lieu de NNUE-vs-exact sur un même
//! réseau, le réseau A (--net) affronte le réseau B (--net-b), TOUS DEUX par
//! le chemin NNUE normal (`utilise_nnue = true`), mêmes budgets --nodes, TT
//! fraîche et de même taille de chaque côté, mêmes paires d'ouvertures jouées
//! une fois de chaque couleur. Les schémas de A et B peuvent différer (chaque
//! moteur porte son propre réseau, aucune vérification croisée). CSV :
//! `net_a,net_b,partie,couleur_a,resultat` (du point de vue de A).
//!
//! Chantier int8 (couperets du chemin quantizé, src/quant.rs) :
//! - `--quant-a` / `--quant-b` (drapeaux sans valeur) activent l'évaluation
//!   int8 pour le camp A / le camp B — prioritaire sur la voie f32 du camp.
//!   Fidélité à budget de nœuds égal : `--net X --net-b X --quant-a` (les
//!   deux camps portent le même réseau, A le lit en int8, B en f32).
//! - `--movetime <ms>` : duel à TEMPS égal par coup au lieu de nœuds (les
//!   budgets --nodes sont ignorés) — c'est là que la vitesse int8 doit se
//!   convertir en force. NB : à temps égal, préférer `--threads 1` (ou
//!   accepter la contention symétrique du pool rayon).

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
    /// Réseau B du duel A/B ; None = mode historique NNUE-vs-exact.
    net_b: Option<String>,
    games: usize,
    nodes: u64,
    /// > 0 : duel à TEMPS égal par coup (ms), --nodes ignoré.
    movetime: u64,
    /// Évaluation quantizée int8 pour le camp A / le camp B.
    quant_a: bool,
    quant_b: bool,
    /// Threads de recherche (lazy SMP) du camp A / du camp B (défaut 1 :
    /// chemin mono-thread historique). Couperet de force SMP : A à N threads
    /// contre B à 1 thread, même movetime, avec --threads 1 (parties
    /// séquentielles, tout le CPU au camp au trait).
    smp_a: u32,
    smp_b: u32,
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
        net_b: None,
        games: 200,
        nodes: 8000,
        movetime: 0,
        quant_a: false,
        quant_b: false,
        smp_a: 1,
        smp_b: 1,
        threads: 4,
        out: "forensic.csv".to_string(),
        seed: 0xF0E5_1C42,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        // Drapeaux SANS valeur : n'avancent que d'un cran.
        match nom.as_str() {
            "--quant-a" => {
                opt.quant_a = true;
                i += 1;
                continue;
            }
            "--quant-b" => {
                opt.quant_b = true;
                i += 1;
                continue;
            }
            _ => {}
        }
        match nom.as_str() {
            "--net" => opt.net = valeur(&args, i, &nom),
            "--net-b" => opt.net_b = Some(valeur(&args, i, &nom)),
            "--games" => opt.games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--nodes" => opt.nodes = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--movetime" => opt.movetime = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--smp-a" => opt.smp_a = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--smp-b" => opt.smp_b = parse_valeur(&valeur(&args, i, &nom), &nom),
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
/// change (f32 incrémental, forward exact, ou int8 quantizé). C'est TOUTE la
/// différence entre les camps.
fn chercheur(net: &Arc<Mlp>, incremental: bool, quant: bool, smp: u32) -> Recherche {
    let mut r = Recherche::new(net.clone(), TAILLE_TT_LOG2);
    r.utilise_nnue = incremental;
    r.utilise_int8 = quant;
    r.threads = smp.max(1);
    r
}

/// Une partie : camp A contre camp B, chacun avec SON réseau et SA voie
/// d'évaluation, mêmes `limites` par coup (nœuds ou movetime), température 0
/// (coup du chercheur tel quel, déterministe). Chercheurs FRAIS (TT vierges
/// de même taille, équité). Arbitrage identique à partie_gating de train.rs.
/// Renvoie les points du camp A (1.0 victoire, 0.5 nulle, 0.0 défaite).
///
/// Mode historique : A = (net, incrémental), B = (même net, forward exact).
/// Mode duel A/B : A = (net_a, incrémental), B = (net_b, incrémental).
/// --quant-a / --quant-b : le camp correspondant évalue en int8.
#[allow(clippy::too_many_arguments)]
fn partie(
    net_a: &Arc<Mlp>,
    incr_a: bool,
    quant_a: bool,
    smp_a: u32,
    net_b: &Arc<Mlp>,
    incr_b: bool,
    quant_b: bool,
    smp_b: u32,
    a_blanc: bool,
    ouverture: &[Move],
    limites: Limites,
) -> f32 {
    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    for m in ouverture {
        pos = pos.play(m).expect("coup d'ouverture légal");
        *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
    }
    let mut camp_a = chercheur(net_a, incr_a, quant_a, smp_a);
    let mut camp_b = chercheur(net_b, incr_b, quant_b, smp_b);
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
        let tour_a = (pos.turn() == Color::White) == a_blanc;
        let res = if tour_a {
            camp_a.cherche(&pos, limites)
        } else {
            camp_b.cherche(&pos, limites)
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
    let cote = if a_blanc { resultat_blancs } else { -resultat_blancs };
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
    // Mode duel A/B : B porte son propre réseau (schéma éventuellement
    // différent de A, aucune vérification croisée) et joue par le chemin NNUE
    // normal, comme A. Mode historique : B est le même réseau que A, évalué
    // par le forward exact.
    let duel_ab = opt.net_b.is_some();
    let (net_b, incr_b) = match &opt.net_b {
        Some(chemin) => (
            Arc::new(Mlp::load(chemin).unwrap_or_else(|e| {
                eprintln!("--net-b {chemin} : chargement impossible ({e})");
                std::process::exit(1);
            })),
            true,
        ),
        None => (net.clone(), false),
    };
    // Budget par coup : temps égal (--movetime) prioritaire sur les nœuds.
    let limites = if opt.movetime > 0 {
        Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: opt.movetime }
    } else {
        Limites { max_noeuds: opt.nodes, max_profondeur: 0, movetime_ms: 0 }
    };
    let budget = if opt.movetime > 0 {
        format!("movetime={} ms", opt.movetime)
    } else {
        format!("nodes={}", opt.nodes)
    };
    match &opt.net_b {
        Some(chemin) => println!(
            "forensic duel A/B : net_a={} sizes_a={:?} quant_a={} smp_a={} net_b={} \
             sizes_b={:?} quant_b={} smp_b={} games={} {budget} threads={} out={} seed={}",
            opt.net, net.sizes, opt.quant_a, opt.smp_a, chemin, net_b.sizes, opt.quant_b,
            opt.smp_b, opt.games, opt.threads, opt.out, opt.seed
        ),
        None => println!(
            "forensic : net={} sizes={:?} quant_a={} quant_b={} games={} {budget} \
             threads={} out={} seed={}",
            opt.net, net.sizes, opt.quant_a, opt.quant_b, opt.games, opt.threads, opt.out,
            opt.seed
        ),
    }

    let paires = opt.games / 2;
    if paires == 0 {
        eprintln!("--games {} : il faut au moins une paire (2 parties)", opt.games);
        std::process::exit(2);
    }

    // Paires en parallèle, une par tâche rayon (même remarque que le gating :
    // sans with_max_len(1), les paquets séquentiels sous-occupent le pool).
    let faites = AtomicUsize::new(0);
    // (indice de partie, couleur du camp A, points du camp A) pour le CSV.
    let resultats: Vec<(usize, &'static str, f32)> = (0..paires)
        .into_par_iter()
        .with_max_len(1)
        .flat_map(|p| {
            // Ouverture aléatoire partagée par la paire, jouée des deux
            // couleurs (A blancs puis A noirs : symétrie exacte, l'outil
            // mesure une asymétrie et ne doit pas en introduire une).
            // Jamais à court de coups en 4 plis.
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
            let pts_blanc = partie(
                &net, true, opt.quant_a, opt.smp_a, &net_b, incr_b, opt.quant_b, opt.smp_b, true,
                &ouverture, limites,
            );
            let pts_noir = partie(
                &net, true, opt.quant_a, opt.smp_a, &net_b, incr_b, opt.quant_b, opt.smp_b, false,
                &ouverture, limites,
            );
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
        if duel_ab {
            writeln!(f, "net_a,net_b,partie,couleur_a,resultat").expect("écriture CSV");
        } else {
            writeln!(f, "net,partie,couleur_nnue,resultat").expect("écriture CSV");
        }
    }
    for (idx, couleur, pts) in &resultats {
        if let Some(chemin_b) = &opt.net_b {
            writeln!(f, "{},{},{},{},{}", opt.net, chemin_b, idx, couleur, pts)
                .expect("écriture CSV");
        } else {
            writeln!(f, "{},{},{},{}", opt.net, idx, couleur, pts).expect("écriture CSV");
        }
    }

    // Récapitulatif : points du camp A (NNUE incrémental en mode historique,
    // réseau --net en mode duel A/B).
    let total = resultats.len() as f32;
    let pts_a: f32 = resultats.iter().map(|(_, _, p)| p).sum();
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
    if duel_ab {
        println!(
            "forensic duel A/B : A {pts_a:.1} / {total:.0} ({:.1} %) — V {v} / N {n} / D {d} \
             (B : {:.1} pts, {:.1} %)",
            100.0 * pts_a / total,
            total - pts_a,
            100.0 * (total - pts_a) / total,
        );
        println!(
            "interpretation : harnais neutre, ouvertures appariees jouees des deux \
             couleurs ; l'ecart a 50 % mesure la difference de force reelle entre A et B."
        );
    } else {
        println!(
            "forensic : NNUE {pts_a:.1} / {total:.0} ({:.1} %) — V {v} / N {n} / D {d} \
             (exact : {:.1} pts, {:.1} %)",
            100.0 * pts_a / total,
            total - pts_a,
            100.0 * (total - pts_a) / total,
        );
        println!(
            "interpretation : ~50 % attendu si l'incremental est fidele ; un ecart \
             significatif a nombre de noeuds fixe prouve des evaluations faussees."
        );
    }
}
