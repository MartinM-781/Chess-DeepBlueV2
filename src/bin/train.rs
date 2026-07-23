//! Entraîneur self-play. Contrat :
//!
//! Options (parse maison sur std::env::args, pas de clap) :
//!   --out models          dossier des modèles/état/métriques
//!   --threads 10          threads rayon (ThreadPoolBuilder global)
//!   --games-per-cycle 128 parties de self-play par cycle
//!   --temperature 0.35    température d'exploration du self-play
//!   --lr 0.001            taux d'apprentissage Adam
//!   --eval-games 120      parties d'évaluation par adversaire de référence
//!   --replay 1200000      positions gardées dans le tampon de rejeu (0 = désactivé)
//!   --elo-every 15        estimation Elo tous les N cycles (0 = désactivée)
//!   --elo-games 24        parties par ancre de l'échelle Elo
//!   --seed 0
//!
//! Boucle par cycle :
//!   1. reprend models/chess_latest.bin + state.json s'ils existent (sinon réseau
//!      neuf, graine --seed) ;
//!   2. self-play : `games_per_cycle` parties en parallèle (rayon,
//!      selfplay::play_training_game, graine dérivée seed+games déjà joués) ;
//!   3. apprentissage : mélange toutes les positions du cycle, minibatchs de 256,
//!      1 époque, nn::train_batch, loss moyenne ; puis, si le tampon de rejeu est
//!      actif, autant de minibatchs supplémentaires échantillonnés uniformément
//!      dans le tampon (chaque position est ainsi revue plusieurs fois au fil des
//!      cycles — meilleure efficacité d'échantillon, courbe plus stable ; le
//!      tampon repart vide à chaque reprise du process, c'est accepté) ;
//!   4. évaluation : NetBot (temperature 0, depth 1) contre RandomBot et contre
//!      MaterialBot(depth 2) via arena::score, `eval_games` parties chacun ;
//!   5. état : trained_secs += durée du cycle (mesurée), games, positions, cycles ;
//!      sauvegarde chess_latest.bin + state.json ; si un palier d'heures cumulées
//!      est franchi (checkpoints::milestone_crossed), copie vers chess_tXh.bin ;
//!   6. métriques : append dans models/metrics.csv, entête si fichier neuf :
//!      `elapsed_hours,games,positions,loss,pct_vs_random,pct_vs_material`
//!      (pourcentages dans [0,100], 1 décimale ; elapsed_hours 3 décimales) ;
//!   7. affiche une ligne de progression flushée :
//!      `[c123] 2.451 h | 12345 parties | 987654 positions | loss 0.812 | vs alea 87.5 % | vs materiel 41.2 %`
//!
//! Boucle infinie (Ctrl-C pour arrêter : tout est sauvé à chaque cycle).

use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use shakmaty::{Chess, Move};

use echec::arena;
use echec::bots::{Bot, MaterialBot, NetBot, RandomBot};
use echec::checkpoints::{self, TrainState};
use echec::elo;
use echec::features::N_FEATURES;
use echec::nn::Mlp;
use echec::selfplay::{self, GameRecord};

/// Plis max d'une partie de self-play (au-delà : arbitrage en nulle, comme en arène).
const MAX_PLIES: u32 = 300;
/// Taille des minibatchs d'apprentissage.
const MINIBATCH: usize = 256;
/// Profondeur du MaterialBot de référence en évaluation.
const PROFONDEUR_MATERIEL: u32 = 2;
/// Profondeur du NetBot en évaluation (température 0).
const PROFONDEUR_EVAL: u32 = 1;
/// Profondeur du NetBot pour l'estimation Elo : 2, comme l'IA servie sur le
/// plateau — l'Elo estimé décrit ce que l'utilisateur affronte réellement.
const PROFONDEUR_ELO: u32 = 2;

/// Options de la ligne de commande (défauts du contrat).
struct Options {
    out: String,
    threads: usize,
    games_per_cycle: usize,
    temperature: f32,
    lr: f32,
    eval_games: usize,
    replay: usize,
    elo_every: u64,
    elo_games: usize,
    seed: u64,
}

/// Tampon de rejeu : anneau de positions encodées à capacité fixe.
/// L'écriture écrase les plus anciennes ; l'échantillonnage est uniforme.
struct Rejeu {
    xs: Vec<f32>,
    zs: Vec<f32>,
    capacite: usize,
    tete: usize,
    len: usize,
}

impl Rejeu {
    fn new(capacite: usize) -> Self {
        Rejeu {
            // Croissance paresseuse : on n'alloue les ~773 f32/position qu'au fil
            // des écritures, pas les ~3,7 Go d'un coup au démarrage.
            xs: Vec::new(),
            zs: Vec::new(),
            capacite,
            tete: 0,
            len: 0,
        }
    }

    fn push(&mut self, x: &[f32], z: f32) {
        if self.len < self.capacite {
            self.xs.extend_from_slice(x);
            self.zs.push(z);
            self.len += 1;
        } else {
            let d = self.tete * N_FEATURES;
            self.xs[d..d + N_FEATURES].copy_from_slice(x);
            self.zs[self.tete] = z;
        }
        self.tete = (self.tete + 1) % self.capacite;
    }

    /// Remplit un minibatch échantillonné uniformément dans le tampon.
    fn echantillonne(&self, rng: &mut StdRng, n: usize,
                     lot_xs: &mut Vec<f32>, lot_zs: &mut Vec<f32>) {
        lot_xs.clear();
        lot_zs.clear();
        for _ in 0..n {
            let i = rng.gen_range(0..self.len);
            lot_xs.extend_from_slice(&self.xs[i * N_FEATURES..(i + 1) * N_FEATURES]);
            lot_zs.push(self.zs[i]);
        }
    }
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
        out: "models".to_string(),
        threads: 10,
        games_per_cycle: 128,
        temperature: 0.35,
        lr: 0.001,
        eval_games: 120,
        replay: 1_200_000,
        elo_every: 15,
        elo_games: 24,
        seed: 0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--out" => opt.out = valeur(&args, i, &nom),
            "--threads" => opt.threads = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--games-per-cycle" => {
                opt.games_per_cycle = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--temperature" => opt.temperature = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--lr" => opt.lr = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--eval-games" => opt.eval_games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--replay" => opt.replay = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--elo-every" => opt.elo_every = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--elo-games" => opt.elo_games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                eprintln!(
                    "options : --out --threads --games-per-cycle --temperature --lr \
                     --eval-games --replay --elo-every --elo-games --seed"
                );
                std::process::exit(2);
            }
        }
        i += 2;
    }
    opt
}

/// Mélangeur déterministe (style SplitMix64) pour dériver des graines
/// indépendantes d'une même graine de base.
fn derive_graine(base: u64, sel: u64) -> u64 {
    let mut z = base ^ sel.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Adaptateur possédant : `arena::score` exige des `Box<dyn Bot>` ('static),
/// alors que `NetBot` emprunte le réseau. On garde donc un `Arc<Mlp>` et on
/// délègue chaque coup à un `NetBot` frais, semé par le RNG de la partie
/// (déterministe pour une graine de partie donnée, sans fuite ni unsafe).
struct NetBotPossedant {
    net: Arc<Mlp>,
    rng: StdRng,
    depth: u32,
}

impl NetBotPossedant {
    fn new(net: Arc<Mlp>, graine: u64, depth: u32) -> Self {
        NetBotPossedant {
            net,
            rng: StdRng::seed_from_u64(graine),
            depth,
        }
    }
}

impl Bot for NetBotPossedant {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        let graine_coup: u64 = self.rng.gen();
        let mut bot = NetBot::new(&self.net, graine_coup, 0.0, self.depth);
        bot.choose(pos)
    }
}

fn main() {
    let opt = parse_options();

    fs::create_dir_all(&opt.out).expect("création du dossier --out");
    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .expect("construction du pool rayon global");

    // 1. Reprise : modèle + état cumulés s'ils existent, sinon départ à neuf.
    let chemin_latest = checkpoints::latest_path(&opt.out);
    let net = if Path::new(&chemin_latest).exists() {
        Mlp::load(&chemin_latest).expect("chargement de chess_latest.bin")
    } else {
        Mlp::new(opt.seed)
    };
    let mut net = Arc::new(net);
    let mut etat = TrainState::load(&opt.out);
    if etat.cycles > 0 {
        println!(
            "reprise : {} cycles, {:.3} h, {} parties, {} positions",
            etat.cycles,
            etat.trained_secs / 3600.0,
            etat.games,
            etat.positions
        );
    } else {
        println!("réseau neuf (graine {})", opt.seed);
    }
    let mut rejeu = (opt.replay > 0).then(|| Rejeu::new(opt.replay));
    if let Some(r) = &rejeu {
        println!(
            "tampon de rejeu : {} positions max (~{:.1} Go)",
            r.capacite,
            (r.capacite * N_FEATURES * 4) as f64 / 1e9
        );
    }
    std::io::stdout().flush().ok();

    // Cycles effectués par CE process (l'estimation Elo tourne au premier cycle
    // local — retour immédiat après un lancement — puis tous les elo_every cycles
    // globaux). Son temps n'entre PAS dans trained_secs : les heures des paliers
    // restent du pur temps d'entraînement.
    let mut cycles_locaux: u64 = 0;

    loop {
        let debut_cycle = Instant::now();

        // 2. Self-play : graines dérivées de seed + parties déjà jouées, pour
        //    qu'une reprise continue exactement la séquence de parties.
        let graines: Vec<u64> = (0..opt.games_per_cycle)
            .map(|i| opt.seed.wrapping_add(etat.games).wrapping_add(i as u64))
            .collect();
        let net_ref: &Mlp = &net;
        let parties: Vec<GameRecord> = graines
            .par_iter()
            .map(|&g| selfplay::play_training_game(net_ref, g, opt.temperature, MAX_PLIES))
            .collect();

        // Concatène toutes les positions du cycle.
        let n_positions: usize = parties.iter().map(|p| p.zs.len()).sum();
        let mut xs: Vec<f32> = Vec::with_capacity(n_positions * N_FEATURES);
        let mut zs: Vec<f32> = Vec::with_capacity(n_positions);
        for p in parties {
            xs.extend(p.xs);
            zs.extend(p.zs);
        }

        // 3. Apprentissage : indices mélangés, minibatchs de 256, 1 époque,
        //    dernier lot partiel accepté. Loss = moyenne pondérée par la taille
        //    des lots. (Aucun Arc cloné à ce stade : get_mut réussit.)
        let net_mut = Arc::get_mut(&mut net).expect("réseau encore partagé à l'apprentissage");
        let mut indices: Vec<usize> = (0..n_positions).collect();
        let mut rng_melange =
            StdRng::seed_from_u64(derive_graine(opt.seed.wrapping_add(etat.cycles), 0x5AFF1E));
        indices.shuffle(&mut rng_melange);
        let mut somme_loss = 0.0f64;
        let mut n_vus = 0usize;
        let mut lot_xs: Vec<f32> = Vec::with_capacity(MINIBATCH * N_FEATURES);
        let mut lot_zs: Vec<f32> = Vec::with_capacity(MINIBATCH);
        for lot in indices.chunks(MINIBATCH) {
            lot_xs.clear();
            lot_zs.clear();
            for &i in lot {
                lot_xs.extend_from_slice(&xs[i * N_FEATURES..(i + 1) * N_FEATURES]);
                lot_zs.push(zs[i]);
            }
            let loss_lot = net_mut.train_batch(&lot_xs, &lot_zs, opt.lr);
            somme_loss += loss_lot as f64 * lot.len() as f64;
            n_vus += lot.len();
        }

        // 3 bis. Rejeu : les positions fraîches entrent dans le tampon, puis on
        // rejoue autant de minibatchs, tirés uniformément dans TOUT le tampon
        // (frais + anciens). Chaque position finit donc revue plusieurs fois au
        // fil des cycles avant d'être écrasée.
        if let Some(r) = rejeu.as_mut() {
            for i in 0..n_positions {
                r.push(&xs[i * N_FEATURES..(i + 1) * N_FEATURES], zs[i]);
            }
            if r.len >= MINIBATCH {
                let mut rng_rejeu = StdRng::seed_from_u64(derive_graine(
                    opt.seed.wrapping_add(etat.cycles),
                    0x8E3E0,
                ));
                for _ in 0..n_positions.div_ceil(MINIBATCH) {
                    r.echantillonne(&mut rng_rejeu, MINIBATCH, &mut lot_xs, &mut lot_zs);
                    let loss_lot = net_mut.train_batch(&lot_xs, &lot_zs, opt.lr);
                    somme_loss += loss_lot as f64 * MINIBATCH as f64;
                    n_vus += MINIBATCH;
                }
            }
        }
        let loss = if n_vus > 0 {
            (somme_loss / n_vus as f64) as f32
        } else {
            0.0
        };

        // 4. Évaluation : NetBot (température 0, profondeur 1) contre les deux
        //    références. Tourne dans le même pool rayon global.
        let net_vs_alea = net.clone();
        let pct_alea = arena::score(
            move |g: u64| -> Box<dyn Bot> {
                Box::new(NetBotPossedant::new(net_vs_alea.clone(), g, PROFONDEUR_EVAL))
            },
            |g: u64| -> Box<dyn Bot> { Box::new(RandomBot::new(g)) },
            opt.eval_games,
            derive_graine(opt.seed.wrapping_add(etat.games), 0xA1EA),
        ) * 100.0;
        let net_vs_materiel = net.clone();
        let pct_materiel = arena::score(
            move |g: u64| -> Box<dyn Bot> {
                Box::new(NetBotPossedant::new(
                    net_vs_materiel.clone(),
                    g,
                    PROFONDEUR_EVAL,
                ))
            },
            |g: u64| -> Box<dyn Bot> { Box::new(MaterialBot::new(g, PROFONDEUR_MATERIEL)) },
            opt.eval_games,
            derive_graine(opt.seed.wrapping_add(etat.games), 0x0A7E),
        ) * 100.0;

        // 5. État cumulé + sauvegardes atomiques (.tmp puis rename).
        let duree_cycle = debut_cycle.elapsed().as_secs_f64();
        let avant_h = etat.trained_secs / 3600.0;
        etat.trained_secs += duree_cycle;
        let apres_h = etat.trained_secs / 3600.0;
        etat.games += opt.games_per_cycle as u64;
        etat.positions += n_positions as u64;
        etat.cycles += 1;

        let chemin_tmp = format!("{chemin_latest}.tmp");
        net.save(&chemin_tmp).expect("écriture du modèle (.tmp)");
        fs::rename(&chemin_tmp, &chemin_latest).expect("renommage du modèle");
        etat.save(&opt.out).expect("écriture de state.json");
        if let Some(h) = checkpoints::milestone_crossed(avant_h, apres_h) {
            let chemin_palier = checkpoints::milestone_path(&opt.out, h);
            fs::copy(&chemin_latest, &chemin_palier).expect("copie de l'instantané de palier");
            println!("palier {h} h franchi -> {chemin_palier}");
        }

        // 6. Métriques : append, entête seulement si le fichier est neuf.
        let chemin_metrics = format!("{}/metrics.csv", opt.out);
        let neuf = !Path::new(&chemin_metrics).exists();
        let mut fichier = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&chemin_metrics)
            .expect("ouverture de metrics.csv");
        if neuf {
            writeln!(
                fichier,
                "elapsed_hours,games,positions,loss,pct_vs_random,pct_vs_material"
            )
            .expect("entête de metrics.csv");
        }
        writeln!(
            fichier,
            "{:.3},{},{},{:.6},{:.1},{:.1}",
            apres_h, etat.games, etat.positions, loss, pct_alea, pct_materiel
        )
        .expect("append dans metrics.csv");

        // 7. Ligne de progression flushée.
        println!(
            "[c{}] {:.3} h | {} parties | {} positions | loss {:.3} | vs alea {:.1} % | vs materiel {:.1} %",
            etat.cycles, apres_h, etat.games, etat.positions, loss, pct_alea, pct_materiel
        );
        std::io::stdout().flush().ok();

        // 8. Estimation Elo : échelle d'ancres + ajustement par maximum de
        //    vraisemblance (voir src/elo.rs). Hors chronométrage des paliers.
        cycles_locaux += 1;
        if opt.elo_every > 0 && (cycles_locaux == 1 || etat.cycles % opt.elo_every == 0) {
            let mesures = elo::mesure(
                &net,
                PROFONDEUR_ELO,
                opt.elo_games,
                derive_graine(opt.seed.wrapping_add(etat.cycles), 0xE10),
            );
            let estimation = elo::ajuste_elo(&mesures);
            let chemin_elo = format!("{}/elo.csv", opt.out);
            let neuf_elo = !Path::new(&chemin_elo).exists();
            let mut fichier_elo = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&chemin_elo)
                .expect("ouverture de elo.csv");
            if neuf_elo {
                writeln!(fichier_elo, "elapsed_hours,elo").expect("entête de elo.csv");
            }
            writeln!(fichier_elo, "{:.3},{:.0}", apres_h, estimation)
                .expect("append dans elo.csv");
            let detail: Vec<String> = mesures
                .iter()
                .map(|m| format!("{} {:.0} %", m.nom, m.score * 100.0))
                .collect();
            println!(
                "  Elo estime ~{:.0} (echelle maison ; {})",
                estimation,
                detail.join(", ")
            );
            std::io::stdout().flush().ok();
        }
    }
}
