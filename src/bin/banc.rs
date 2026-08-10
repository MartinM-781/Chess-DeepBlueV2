//! Banc d'essai par PHASE de partie : combien de centipions le champion
//! perd-il, en moyenne, sur des positions d'ouverture / de transition / de
//! finale ?
//!
//! Pourquoi ce binaire. Le match contre le Fantôme de Deep Blue a livré son
//! diagnostic (arbitre, 327 plis) : perte moyenne 12,5 cp en ouverture,
//! 25,0 en milieu, **51,9 en transition**, 0,1 en finale — huit fois pire que
//! l'adversaire dans la seule phase de transition. Mais cette mesure a coûté un
//! match entier (20 h). Pour piloter un chantier, il faut la même mesure en
//! QUELQUES MINUTES : c'est exactement ce que fait ce banc.
//!
//! Principe, identique à celui de l'arbitre (src/arbitre.rs) mais sur un jeu de
//! positions TIRÉ, donc reproductible et rapide :
//!   1. tirer N positions d'une famille (src/departs.rs : ouverture du livre,
//!      milieu tardif « transition », finale générée) ;
//!   2. y faire jouer le champion à budget fixe (--noeuds) ;
//!   3. demander à un Stockfish PLEINE FORCE l'évaluation avant le coup et
//!      après le coup ; la différence, du point de vue du champion, est la
//!      perte en centipions de ce coup ;
//!   4. rendre la moyenne par famille, plus la répartition des classements
//!      (mêmes seuils que l'arbitre : meilleur / excellent / bon / imprécision
//!      / erreur / gaffe).
//!
//! Le protocole est celui d'un couperet : on mesure AVANT un changement de
//! régime, on mesure APRÈS, et la comparaison tranche. La graine fixe (--seed)
//! garantit que les deux mesures portent sur LES MÊMES positions.
//!
//! Exemple :
//!   banc --net models/chess_best.bin --positions 60 --noeuds 200000 \
//!        --engine engines/stockfish/stockfish-windows-x86-64-avx2.exe \
//!        --movetime 2000 --int8 --syzygy engines/syzygy --csv models/banc.csv

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use shakmaty::fen::Fen;
use shakmaty::{Chess, EnPassantMode, Position};

use echec::arbitre::{classement, cp_du_score};
use echec::departs;
use echec::nn::Mlp;
use echec::search::{Limites, Recherche};
use echec::uci::UciEngine;

struct Opt {
    net: String,
    engine: String,
    positions: usize,
    noeuds: u64,
    movetime: u64,
    threads: u32,
    int8: bool,
    syzygy: String,
    seed: u64,
    csv: String,
    tt_log2: u32,
}

fn valeur(args: &[String], i: usize, nom: &str) -> String {
    args.get(i + 1).cloned().unwrap_or_else(|| {
        eprintln!("option {nom} : valeur manquante");
        std::process::exit(2);
    })
}

fn parse<T: std::str::FromStr>(v: &str, nom: &str) -> T {
    v.parse().unwrap_or_else(|_| {
        eprintln!("option {nom} : valeur illisible « {v} »");
        std::process::exit(2);
    })
}

fn parse_options() -> Opt {
    let mut opt = Opt {
        net: "models/chess_best.bin".to_string(),
        engine: "engines/stockfish/stockfish-windows-x86-64-avx2.exe".to_string(),
        positions: 40,
        noeuds: 200_000,
        movetime: 2000,
        threads: 1,
        int8: false,
        syzygy: String::new(),
        seed: 20_260_809,
        csv: String::new(),
        tt_log2: 22,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        if nom == "--int8" {
            opt.int8 = true;
            i += 1;
            continue;
        }
        match nom.as_str() {
            "--net" => opt.net = valeur(&args, i, &nom),
            "--engine" => opt.engine = valeur(&args, i, &nom),
            "--positions" => opt.positions = parse(&valeur(&args, i, &nom), &nom),
            "--noeuds" => opt.noeuds = parse(&valeur(&args, i, &nom), &nom),
            "--movetime" => opt.movetime = parse(&valeur(&args, i, &nom), &nom),
            "--threads" => opt.threads = parse(&valeur(&args, i, &nom), &nom),
            "--syzygy" => opt.syzygy = valeur(&args, i, &nom),
            "--seed" => opt.seed = parse(&valeur(&args, i, &nom), &nom),
            "--csv" => opt.csv = valeur(&args, i, &nom),
            "--tt-log2" => opt.tt_log2 = parse(&valeur(&args, i, &nom), &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                eprintln!(
                    "usage : banc [--net <bin>] [--engine <exe>] [--positions 40] \
                     [--noeuds 200000] [--movetime 2000] [--threads 1] [--int8] \
                     [--syzygy <dir>] [--seed S] [--csv <fichier>] [--tt-log2 22]"
                );
                std::process::exit(2);
            }
        }
        i += 2;
    }
    opt
}

/// Les trois familles mesurables telles quelles par `departs` : chaque entrée
/// donne (nom, p_ouverture, p_finale, p_transition) — une seule à 1.0.
const FAMILLES: &[(&str, f32, f32, f32)] = &[
    ("ouverture", 1.0, 0.0, 0.0),
    ("transition", 0.0, 0.0, 1.0),
    ("finale", 0.0, 1.0, 0.0),
];

fn main() {
    let opt = parse_options();
    echec::pleine_puissance();

    let net = Arc::new(Mlp::load(&opt.net).unwrap_or_else(|e| {
        eprintln!("chargement du réseau {} : {e}", opt.net);
        std::process::exit(1);
    }));
    println!(
        "banc : {} {:?} | {} positions/famille | {} nœuds/coup | arbitre {} à {} ms",
        opt.net,
        net.sizes,
        opt.positions,
        opt.noeuds,
        opt.engine,
        opt.movetime
    );

    let mut recherche = Recherche::new(net.clone(), opt.tt_log2);
    recherche.threads = opt.threads;
    recherche.utilise_int8 = opt.int8;
    if !opt.syzygy.is_empty() {
        match echec::syzygy::charge(&opt.syzygy) {
            Ok((tables, n)) => {
                println!("syzygy : {n} tables");
                recherche.syzygy = Some(Arc::new(tables));
            }
            Err(e) => eprintln!("--syzygy {} : {e} (banc sans tables)", opt.syzygy),
        }
    }
    let mut moteur = UciEngine::lance_pleine_force(&opt.engine, opt.movetime as u32)
        .unwrap_or_else(|e| {
            eprintln!("lancement du moteur arbitre : {e}");
            std::process::exit(1);
        });

    let limites = Limites { max_noeuds: opt.noeuds, max_profondeur: 0, movetime_ms: 0 };
    let mut lignes_csv: Vec<String> = Vec::new();
    println!();

    for (famille, p_o, p_f, p_t) in FAMILLES {
        // Graine DÉRIVÉE du nom de famille : deux exécutions au même --seed
        // rejouent exactement les mêmes positions, famille par famille — c'est
        // ce qui rend deux mesures comparables de part et d'autre d'un
        // changement de régime.
        let mut graine = opt.seed;
        for o in famille.bytes() {
            graine = graine.wrapping_mul(0x0100_0000_01B3) ^ o as u64;
        }
        let mut rng = StdRng::seed_from_u64(graine);
        let mut pertes: Vec<i32> = Vec::new();
        let mut compte: std::collections::BTreeMap<&'static str, usize> = Default::default();
        let mut sautees = 0usize;

        for k in 0..opt.positions {
            let depart = departs::tirage_complet(&mut rng, *p_o, *p_f, *p_t);
            let pos: Chess = depart.pos;
            if pos.legal_moves().is_empty() || pos.is_game_over() {
                sautees += 1;
                continue;
            }
            let fen_avant = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
            // 1. Évaluation de référence AVANT le coup (point de vue du trait,
            //    c'est-à-dire du champion, qui joue).
            let Ok((_, Some(s_avant))) =
                moteur.meilleur_coup_et_score_brut_fen(&fen_avant, opt.movetime)
            else {
                sautees += 1;
                continue;
            };
            let avant = cp_du_score(s_avant);
            // Positions DÉJÀ TRANCHÉES (mat annoncé ou plus de 8 pions
            // d'écart) : la qualité du coup n'y veut plus rien dire — on gagne
            // ou on perd de toute façon —, et une seule d'entre elles suffit à
            // écraser la moyenne (mesuré : 1568 cp de moyenne en finale pour
            // une médiane de 0). Elles sont écartées du banc.
            if avant.abs() > 800 {
                sautees += 1;
                continue;
            }

            // 2. Le coup du champion.
            recherche.nouvelle_partie();
            let res = recherche.cherche(&pos, limites);
            let Some(coup) = res.coup else {
                sautees += 1;
                continue;
            };
            let mut apres_pos = pos.clone();
            apres_pos.play_unchecked(&coup);

            // 3. Évaluation APRÈS le coup. Le trait a changé : le score rendu
            //    est celui de l'ADVERSAIRE, donc l'opposé du nôtre.
            let perte = if apres_pos.is_game_over() {
                // Mat ou pat immédiat : rien à demander au moteur.
                0
            } else {
                let fen_apres =
                    Fen::from_position(apres_pos.clone(), EnPassantMode::Legal).to_string();
                let Ok((_, Some(s_apres))) =
                    moteur.meilleur_coup_et_score_brut_fen(&fen_apres, opt.movetime)
                else {
                    sautees += 1;
                    continue;
                };
                let apres_nous = -cp_du_score(s_apres);
                // Perte plafonnée : au-delà d'une pièce et demie, le coup est
                // déjà catastrophique et la valeur exacte n'ajoute rien — sans
                // ce plafond, un mat encaissé (±32000) écrase toute la série.
                (avant - apres_nous).clamp(0, 600)
            };

            let cl = classement(perte, false).nom();
            *compte.entry(cl).or_insert(0) += 1;
            pertes.push(perte);
            if !opt.csv.is_empty() {
                lignes_csv.push(format!(
                    "{famille},{k},{avant},{perte},{cl},\"{fen_avant}\""
                ));
            }
            if (k + 1) % 10 == 0 {
                print!("\r  {famille} : {}/{} positions", k + 1, opt.positions);
                use std::io::Write as _;
                std::io::stdout().flush().ok();
            }
        }

        let moyenne = if pertes.is_empty() {
            f64::NAN
        } else {
            pertes.iter().map(|&p| p as f64).sum::<f64>() / pertes.len() as f64
        };
        let mediane = {
            let mut v = pertes.clone();
            v.sort_unstable();
            v.get(v.len() / 2).copied().unwrap_or(0)
        };
        println!(
            "\r  {famille:11} perte moyenne {moyenne:6.1} cp | médiane {mediane:4} cp | \
             {} positions{}",
            pertes.len(),
            if sautees > 0 { format!(" ({sautees} sautée(s))") } else { String::new() }
        );
        let detail: Vec<String> =
            compte.iter().map(|(cl, n)| format!("{n} {cl}")).collect();
        println!("              {}", detail.join(", "));
    }

    if !opt.csv.is_empty() {
        let existe = std::path::Path::new(&opt.csv).exists();
        let mut contenu = String::new();
        if !existe {
            contenu.push_str("famille,indice,eval_avant_cp,perte_cp,classement,fen\n");
        }
        contenu.push_str(&lignes_csv.join("\n"));
        contenu.push('\n');
        use std::io::Write as _;
        match std::fs::OpenOptions::new().create(true).append(true).open(&opt.csv) {
            Ok(mut f) => {
                let _ = f.write_all(contenu.as_bytes());
                println!("\njournal : {}", opt.csv);
            }
            Err(e) => eprintln!("écriture {} : {e}", opt.csv),
        }
    }
}
