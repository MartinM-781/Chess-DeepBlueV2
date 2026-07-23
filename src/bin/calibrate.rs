//! Calibration de l'échelle Elo maison contre Stockfish à force limitée.
//!
//! Principe : le réseau courant (models/chess_latest.bin) affronte
//!   1. les ancres maison (elo::mesure — mêmes duels que l'entraîneur),
//!   2. Stockfish UCI_Elo à un ou plusieurs paliers (--elos),
//! puis TOUTES les mesures alimentent le même ajustement par maximum de
//! vraisemblance (elo::ajuste_elo). Les paliers Stockfish étant des Elo réels,
//! ils recalent l'échelle maison sur une référence externe.
//!
//! Options (parse maison, comme train.rs) :
//!   --games 4        parties par ancre maison ET par palier Stockfish
//!   --movetime 30    ms de réflexion Stockfish par coup
//!   --elos 1320      paliers UCI_Elo, séparés par des virgules (ex. 1320,1700)
//!   --depth 2        profondeur du NetBot (2 = comme l'IA servie)
//!   --out models     dossier modèles (entrée chess_latest.bin, sortie elo_calib.csv)
//!   --engine engines/stockfish/stockfish-windows-x86-64-avx2.exe
//!   --threads 4      threads rayon (parties parallèles → autant de processus moteur)
//!   --seed 0
//!
//! Sortie : une ligne par mesure (ancres + stockfish-XXXX), l'« Elo calibre »,
//! et une ligne ajoutée à models/elo_calib.csv.

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use shakmaty::{Chess, Move};

use echec::arena;
use echec::bots::{Bot, NetBot};
use echec::checkpoints;
use echec::elo::{self, MesureAncre};
use echec::nn::Mlp;
use echec::uci::{StockfishBot, UciEngine};

/// Options de la ligne de commande (défauts ci-dessus).
struct Options {
    games: usize,
    movetime: u64,
    elos: Vec<u32>,
    depth: u32,
    out: String,
    engine: String,
    threads: usize,
    seed: u64,
}

/// Valeur suivant l'option `nom`, ou sortie propre si elle manque.
fn valeur(args: &[String], i: usize, nom: &str) -> String {
    args.get(i + 1).cloned().unwrap_or_else(|| {
        eprintln!("option {nom} : valeur manquante");
        std::process::exit(2);
    })
}

/// Parse une valeur d'option, ou sortie propre si elle est invalide.
fn parse_valeur<T: std::str::FromStr>(s: &str, nom: &str) -> T {
    s.parse().unwrap_or_else(|_| {
        eprintln!("option {nom} : valeur invalide « {s} »");
        std::process::exit(2);
    })
}

fn parse_options() -> Options {
    let mut opt = Options {
        games: 4,
        movetime: 30,
        elos: vec![1320],
        depth: 2,
        out: "models".to_string(),
        engine: "engines/stockfish/stockfish-windows-x86-64-avx2.exe".to_string(),
        threads: 4,
        seed: 0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--games" => opt.games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--movetime" => opt.movetime = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--elos" => {
                opt.elos = valeur(&args, i, &nom)
                    .split(',')
                    .map(|s| parse_valeur(s.trim(), &nom))
                    .collect();
            }
            "--depth" => opt.depth = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--out" => opt.out = valeur(&args, i, &nom),
            "--engine" => opt.engine = valeur(&args, i, &nom),
            "--threads" => opt.threads = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                eprintln!(
                    "options : --games --movetime --elos --depth --out --engine --threads --seed"
                );
                std::process::exit(2);
            }
        }
        i += 2;
    }
    if opt.games == 0 || opt.elos.is_empty() {
        eprintln!("--games doit être > 0 et --elos non vide");
        std::process::exit(2);
    }
    opt
}

/// Mélangeur déterministe (même SplitMix64 que train.rs) pour dériver des
/// graines indépendantes de la graine de base.
fn derive_graine(base: u64, sel: u64) -> u64 {
    let mut z = base ^ sel.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Adaptateur possédant (même astuce que NetBotPossedant de train.rs) :
/// arena::score exige des Box<dyn Bot> 'static, NetBot emprunte le réseau —
/// on garde un Arc<Mlp> et on délègue chaque coup à un NetBot frais.
struct NetBotPossedant {
    net: Arc<Mlp>,
    rng: StdRng,
    depth: u32,
}

impl NetBotPossedant {
    fn new(net: Arc<Mlp>, graine: u64, depth: u32) -> Self {
        NetBotPossedant { net, rng: StdRng::seed_from_u64(graine), depth }
    }
}

impl Bot for NetBotPossedant {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        let graine_coup: u64 = self.rng.gen();
        NetBot::new(&self.net, graine_coup, 0.0, self.depth).choose(pos)
    }
}

fn main() {
    let opt = parse_options();
    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .expect("construction du pool rayon global");

    // Modèle courant (celui que l'entraîneur sauve à chaque cycle).
    let chemin_modele = checkpoints::latest_path(&opt.out);
    let net = Arc::new(Mlp::load(&chemin_modele).unwrap_or_else(|e| {
        eprintln!("chargement de {chemin_modele} : {e}");
        std::process::exit(1);
    }));
    println!(
        "calibration : {} | depth {} | {} parties/ancre | movetime {} ms",
        chemin_modele, opt.depth, opt.games, opt.movetime
    );

    // Sonde UCI : vérifie le moteur et relève les bornes UCI_Elo AVANT de
    // jouer, pour nommer les ancres avec l'Elo effectivement appliqué (clamp).
    let sonde = UciEngine::lance(&opt.engine).unwrap_or_else(|e| {
        eprintln!("lancement du moteur {} : {e}", opt.engine);
        std::process::exit(1);
    });
    let (elo_min, elo_max) = (sonde.elo_min, sonde.elo_max);
    drop(sonde); // quitte proprement (Drop) : pas de zombie
    println!("moteur : {} (UCI_Elo {}..{})", opt.engine, elo_min, elo_max);

    // 1. Ancres maison : mêmes duels que l'entraîneur (elo::mesure).
    let mut mesures = elo::mesure(
        &net,
        opt.depth,
        opt.games,
        derive_graine(opt.seed, 0xCA11B),
    );

    // 2. Paliers Stockfish : arena::score, couleurs alternées, un processus
    //    moteur par partie (nécessaire au parallélisme, tués par Drop).
    for (k, &elo_demande) in opt.elos.iter().enumerate() {
        let elo_reel = elo_demande.clamp(elo_min, elo_max);
        if elo_reel != elo_demande {
            println!("  UCI_Elo {elo_demande} clampé à {elo_reel}");
        }
        let net_a = net.clone();
        let depth = opt.depth;
        let engine = opt.engine.clone();
        let movetime = opt.movetime;
        let score = arena::score(
            move |g: u64| -> Box<dyn Bot> {
                Box::new(NetBotPossedant::new(net_a.clone(), g, depth))
            },
            |_g: u64| -> Box<dyn Bot> {
                Box::new(
                    StockfishBot::new(&engine, elo_reel, movetime)
                        .unwrap_or_else(|e| panic!("lancement Stockfish : {e}")),
                )
            },
            opt.games,
            derive_graine(opt.seed, 0x5F00 + k as u64),
        ) as f64;
        // Nom 'static : fuite volontaire et bornée (binaire court, un nom par palier).
        let nom: &'static str = Box::leak(format!("stockfish-{elo_reel}").into_boxed_str());
        mesures.push(MesureAncre {
            nom,
            elo_ancre: elo_reel as f64,
            score,
            parties: opt.games,
        });
    }

    // 3. Ajustement MLE sur TOUTES les mesures (maison + Stockfish).
    for m in &mesures {
        println!(
            "  {:<16} (Elo {:>4.0}) : {:>5.1} % sur {} parties",
            m.nom,
            m.elo_ancre,
            m.score * 100.0,
            m.parties
        );
    }
    let estimation = elo::ajuste_elo(&mesures);
    println!("Elo calibre : {estimation:.0}");

    // 4. Journal CSV : une ligne par calibration, entête si fichier neuf.
    //    Détail encodé nom=score avec « ; » (pas de virgule parasite dans la
    //    colonne) pour garder le CSV bien formé.
    let horodatage = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let detail: Vec<String> = mesures
        .iter()
        .map(|m| format!("{}={:.3}", m.nom.replace([',', ';'], "_"), m.score))
        .collect();
    let chemin_csv = format!("{}/elo_calib.csv", opt.out);
    let neuf = !Path::new(&chemin_csv).exists();
    let mut fichier = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&chemin_csv)
        .expect("ouverture de elo_calib.csv");
    if neuf {
        writeln!(fichier, "horodatage_unix,elo_calibre,parties,movetime_ms,depth,detail")
            .expect("entête de elo_calib.csv");
    }
    writeln!(
        fichier,
        "{},{:.0},{},{},{},{}",
        horodatage,
        estimation,
        opt.games,
        opt.movetime,
        opt.depth,
        detail.join(";")
    )
    .expect("append dans elo_calib.csv");
    println!("journal : {chemin_csv}");
}
