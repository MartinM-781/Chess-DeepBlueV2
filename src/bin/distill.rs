//! Distillation : un réseau élève (architecture élargie, ex. [773,1024,128,1])
//! apprend à IMITER le champion actuel (le prof) — même entrée → même sortie.
//! On clone la FONCTION d'évaluation, pas la politique : aucune recherche,
//! l'étiquette de chaque position est simplement prof.forward_one(encode(pos)).
//! L'élève ainsi amorcé sert ensuite de point de départ au TD-leaf.
//!
//! Options (parse maison sur std::env::args, comme train/serve) :
//!   --teacher models/chess_best.bin   modèle du prof (Mlp::load)
//!   --sizes 773,1024,128,1            architecture de l'élève (entiers, virgules)
//!   --positions 600000                taille du corpus de distillation
//!   --lr 0.001                        taux d'apprentissage Adam
//!   --seed 0
//!   --out models/distill_student.bin  destination de l'élève
//!
//! Déroulé :
//!   1. corpus : parties de diversification jouées par NetBot(prof, T=0.6, 1 pli)
//!      entrecoupées de coups aléatoires (~1 sur 6) pour couvrir large ; mêmes
//!      règles de nulle qu'en arène (pat, matériel, 50 coups, 3 répétitions,
//!      300 plis max) ; génération parallèle rayon (with_max_len(1)) ;
//!   2. apprentissage : mélange, minibatchs 256, jusqu'à 6 époques, arrêt
//!      anticipé si la loss moyenne d'une époque passe sous 0.0004 ;
//!   3. validation sur 20 000 positions FRAÎCHES (autres graines) : MSE
//!      élève-vs-prof et % de positions où |élève - prof| > 0.1 ;
//!   4. sauvegarde vers --out ; code de sortie 0 seulement si MSE < 0.002.

use std::collections::HashMap;
use std::io::Write as _;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, EnPassantMode, Position};

use echec::bots::{Bot, NetBot, RandomBot};
use echec::features::{encode, N_FEATURES};
use echec::nn::Mlp;

/// Plis maximum d'une partie de diversification (comme en arène).
const MAX_PLIS: u32 = 300;
/// Température du NetBot prof pendant la génération (1 pli, softmax).
const TEMPERATURE_DIVERSIFICATION: f32 = 0.6;
/// Un coup sur RATIO_ALEA (en moyenne) est joué par le RandomBot.
const RATIO_ALEA: u64 = 6;
/// Taille des minibatchs d'apprentissage.
const MINIBATCH: usize = 256;
/// Nombre maximal d'époques sur le corpus.
const MAX_EPOQUES: usize = 6;
/// Arrêt anticipé : loss moyenne d'époque sous ce seuil.
const SEUIL_LOSS_EPOQUE: f32 = 0.0004;
/// Positions fraîches de validation.
const N_VALIDATION: usize = 20_000;
/// Seuil de réussite final : MSE élève-vs-prof.
const SEUIL_MSE_VALIDATION: f32 = 0.002;
/// Estimation grossière de positions par partie (pour dimensionner les vagues
/// de génération ; seule la vitesse en dépend, jamais la correction).
const POSITIONS_PAR_PARTIE_ESTIMEES: usize = 60;
/// Décalage de graines entre corpus d'apprentissage et corpus de validation :
/// les deux ne partagent AUCUNE graine de partie.
const DECALAGE_GRAINES_VALIDATION: u64 = 0x9E37_79B9_7F4A_7C15;

/// Options de la ligne de commande (défauts du contrat).
struct Options {
    teacher: String,
    sizes: Vec<usize>,
    positions: usize,
    lr: f32,
    seed: u64,
    out: String,
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

/// Parse une liste d'entiers séparés par des virgules (ex. « 773,1024,128,1 »).
fn parse_tailles(s: &str, nom: &str) -> Vec<usize> {
    s.split(',').map(|morceau| parse_valeur(morceau.trim(), nom)).collect()
}

fn parse_options() -> Options {
    let mut opt = Options {
        teacher: "models/chess_best.bin".to_string(),
        sizes: vec![N_FEATURES, 1024, 128, 1],
        positions: 600_000,
        lr: 0.001,
        seed: 0,
        out: "models/distill_student.bin".to_string(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--teacher" => opt.teacher = valeur(&args, i, &nom),
            "--sizes" => opt.sizes = parse_tailles(&valeur(&args, i, &nom), &nom),
            "--positions" => opt.positions = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--lr" => opt.lr = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--out" => opt.out = valeur(&args, i, &nom),
            autre => {
                eprintln!("option inconnue : {autre}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    opt
}

fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Joue UNE partie de diversification et renvoie (features concaténées,
/// étiquettes du prof). Chaque position visitée (avant le coup) est encodée et
/// étiquetée par prof.forward_one — pas de recherche. Les coups sont ceux du
/// NetBot(prof, T=0.6, 1 pli), sauf ~1 sur 6 tiré au RandomBot pour élargir la
/// couverture. Mêmes règles de nulle qu'en arène, 300 plis max.
fn partie_corpus(prof: &Mlp, graine: u64) -> (Vec<f32>, Vec<f32>) {
    let mut bot_prof = NetBot::new(prof, graine, TEMPERATURE_DIVERSIFICATION, 1);
    let mut bot_alea = RandomBot::new(graine.wrapping_add(0xA5A5_A5A5));
    // Rng du choix « prof ou aléatoire » : indépendant des deux bots.
    let mut rng = StdRng::seed_from_u64(graine.wrapping_add(0x51ED_51ED));

    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    let mut plis = 0u32;

    let mut buf = vec![0.0f32; N_FEATURES];
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    loop {
        // Fins de partie : mat/pat, matériel insuffisant, 50 coups, 300 plis.
        if pos.legal_moves().is_empty()
            || pos.is_insufficient_material()
            || pos.halfmoves() >= 100
            || plis >= MAX_PLIS
        {
            break;
        }

        // Enregistre la position courante : entrée = encode, étiquette = prof.
        encode(&pos, &mut buf);
        ys.push(prof.forward_one(&buf));
        xs.extend_from_slice(&buf);

        // ~1 coup sur 6 : RandomBot ; sinon NetBot du prof (1 pli, T=0.6).
        let coup = if rng.gen_range(0..RATIO_ALEA) == 0 {
            bot_alea.choose(&pos)
        } else {
            bot_prof.choose(&pos)
        }
        .expect("coups légaux non vides");
        pos = pos.play(&coup).expect("coup légal");
        plis += 1;

        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            break; // 3e répétition : nulle
        }
    }
    (xs, ys)
}

/// Génère exactement `n_cible` positions étiquetées, par vagues de parties
/// parallèles (rayon, with_max_len(1) : une partie par tâche, sinon les
/// ouvriers libres n'ont plus rien à voler — voir arena.rs). Les graines de
/// partie sont `graine_base + compteur`, déterministes.
fn genere_corpus(prof: &Mlp, n_cible: usize, graine_base: u64, etiquette: &str) -> (Vec<f32>, Vec<f32>) {
    let mut xs: Vec<f32> = Vec::with_capacity(n_cible * N_FEATURES);
    let mut ys: Vec<f32> = Vec::with_capacity(n_cible);
    let mut compteur_graines = 0u64;

    while ys.len() < n_cible {
        let manque = n_cible - ys.len();
        // Assez de parties pour couvrir le manque (estimation), au moins une
        // vague pleine pour occuper tous les cœurs.
        let n_parties = (manque / POSITIONS_PAR_PARTIE_ESTIMEES + 1).max(rayon::current_num_threads());
        let parties: Vec<(Vec<f32>, Vec<f32>)> = (0..n_parties)
            .into_par_iter()
            .with_max_len(1)
            .map(|i| partie_corpus(prof, graine_base.wrapping_add(compteur_graines + i as u64)))
            .collect();
        compteur_graines += n_parties as u64;

        for (px, py) in parties {
            if ys.len() >= n_cible {
                break;
            }
            let prendre = (n_cible - ys.len()).min(py.len());
            xs.extend_from_slice(&px[..prendre * N_FEATURES]);
            ys.extend_from_slice(&py[..prendre]);
        }
        print!("\r[{etiquette}] {} / {} positions", ys.len(), n_cible);
        let _ = std::io::stdout().flush();
    }
    println!();
    (xs, ys)
}

fn main() {
    echec::pleine_puissance();
    let debut = Instant::now();
    let opt = parse_options();

    // Garde d'architecture : la sortie doit être scalaire (tanh). Vérifié ICI,
    // avant toute génération de corpus, avec le même code de sortie 2 que les
    // autres erreurs d'options — plus lisible que la panique du garde de
    // `Mlp::new_avec_tailles` pour un --sizes tronqué (ex. « 773,1024,128 »).
    if opt.sizes.last() != Some(&1) {
        eprintln!(
            "option --sizes : la dernière couche doit valoir 1 (sortie scalaire tanh), reçu {:?}",
            opt.sizes
        );
        std::process::exit(2);
    }

    // 1. Prof et élève.
    let prof = Mlp::load(&opt.teacher).unwrap_or_else(|e| {
        eprintln!("chargement du prof {} : {e}", opt.teacher);
        std::process::exit(2);
    });
    let mut eleve = Mlp::new_avec_tailles(&opt.sizes, opt.seed);
    println!(
        "distillation : prof {} {:?} -> eleve {:?} | {} positions | lr {} | graine {}",
        opt.teacher, prof.sizes, eleve.sizes, opt.positions, opt.lr, opt.seed
    );

    // 2. Corpus d'apprentissage.
    let (xs, ys) = genere_corpus(&prof, opt.positions, opt.seed, "corpus");

    // 3. Apprentissage : mélange des indices, minibatchs de 256, jusqu'à
    //    6 époques, arrêt anticipé si la loss moyenne d'époque < 0.0004.
    let n = ys.len();
    let mut indices: Vec<usize> = (0..n).collect();
    // Rng du mélange : décalé de la graine des parties (aucun recouvrement).
    let mut rng = StdRng::seed_from_u64(opt.seed.wrapping_add(0x4D45_4C41_4E47_45u64));
    let mut lot_xs: Vec<f32> = Vec::with_capacity(MINIBATCH * N_FEATURES);
    let mut lot_ys: Vec<f32> = Vec::with_capacity(MINIBATCH);
    let mut epoques_faites = 0usize;
    for epoque in 1..=MAX_EPOQUES {
        indices.shuffle(&mut rng);
        let mut somme_loss = 0.0f64;
        let mut n_batchs = 0usize;
        for lot in indices.chunks(MINIBATCH) {
            lot_xs.clear();
            lot_ys.clear();
            for &i in lot {
                lot_xs.extend_from_slice(&xs[i * N_FEATURES..(i + 1) * N_FEATURES]);
                lot_ys.push(ys[i]);
            }
            let loss = eleve.train_batch(&lot_xs, &lot_ys, opt.lr);
            somme_loss += loss as f64;
            n_batchs += 1;
            if n_batchs % 200 == 0 {
                print!(
                    "\r[epoque {epoque}] batch {n_batchs} | loss moyenne {:.6}",
                    somme_loss / n_batchs as f64
                );
                let _ = std::io::stdout().flush();
            }
        }
        let loss_epoque = (somme_loss / n_batchs.max(1) as f64) as f32;
        epoques_faites = epoque;
        println!("\r[epoque {epoque}] {n_batchs} batchs | loss moyenne {loss_epoque:.6}");
        if loss_epoque < SEUIL_LOSS_EPOQUE {
            println!("arret anticipe : loss d'epoque {loss_epoque:.6} < {SEUIL_LOSS_EPOQUE}");
            break;
        }
    }
    drop(xs);
    drop(ys);

    // 4. Validation : positions FRAÎCHES (graines décalées, jamais utilisées
    //    par le corpus d'apprentissage).
    let (val_xs, val_ys) = genere_corpus(
        &prof,
        N_VALIDATION,
        opt.seed.wrapping_add(DECALAGE_GRAINES_VALIDATION),
        "validation",
    );
    // Passe avant de l'élève en parallèle (lecture seule, &self).
    let sorties: Vec<f32> = val_xs
        .par_chunks(N_FEATURES)
        .map(|x| eleve.forward_one(x))
        .collect();
    let mut somme_carres = 0.0f64;
    let mut n_gros_ecarts = 0usize;
    for (s, t) in sorties.iter().zip(&val_ys) {
        let e = (s - t) as f64;
        somme_carres += e * e;
        if (s - t).abs() > 0.1 {
            n_gros_ecarts += 1;
        }
    }
    let mse = (somme_carres / val_ys.len() as f64) as f32;
    let pct_gros_ecarts = 100.0 * n_gros_ecarts as f32 / val_ys.len() as f32;
    println!(
        "validation : MSE eleve-vs-prof {mse:.6} | |ecart| > 0.1 : {pct_gros_ecarts:.2} % ({n_gros_ecarts} / {})",
        val_ys.len()
    );

    // Sauvegarde de l'élève.
    if let Err(e) = eleve.save(&opt.out) {
        eprintln!("sauvegarde de {} : {e}", opt.out);
        std::process::exit(2);
    }
    println!("eleve sauvegarde -> {}", opt.out);

    // 5. Récapitulatif.
    let duree = debut.elapsed().as_secs_f64();
    println!(
        "recap : tailles {:?} | {} positions | {epoques_faites} epoque(s) | MSE finale {mse:.6} | {duree:.1} s",
        eleve.sizes, opt.positions
    );

    // Code de sortie : 0 seulement si la distillation est fidèle.
    std::process::exit(if mse < SEUIL_MSE_VALIDATION { 0 } else { 1 });
}
