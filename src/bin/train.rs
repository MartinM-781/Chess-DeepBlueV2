//! Entraîneur self-play. Contrat :
//!
//! Options (parse maison sur std::env::args, pas de clap) :
//!   --out models          dossier des modèles/état/métriques
//!   --threads 10          threads rayon (ThreadPoolBuilder global)
//!   --games-per-cycle 128 parties de self-play par cycle
//!   --temperature 0.35    température d'exploration du self-play 1-PLI
//!                         (--search-nodes 0) UNIQUEMENT ; en régime recherche
//!                         elle est ignorée : les températures sont figées par
//!                         le contrat OptionsRecherche (0.2, ouverture 0.8)
//!   --lr 0.001            taux d'apprentissage Adam
//!   --eval-games 120      parties d'évaluation par adversaire de référence
//!   --replay 1200000      positions gardées dans le tampon de rejeu (0 = désactivé)
//!   --elo-every 15        estimation Elo tous les N cycles (0 = désactivée)
//!   --elo-games 24        parties par ancre de l'échelle Elo
//!   --seed 0
//!   --search-nodes 1500   nœuds de recherche par coup du self-play
//!                         (0 = ancien régime 1-pli, comportement intact)
//!   --td-lambda 0.3       λ des cibles TD-leaf (régime recherche)
//!   --gate-every 10       gating tous les N cycles en régime recherche
//!                         (0 = pas de gating)
//!   --gate-games 64       parties du duel de gating, jouées par PAIRES à
//!                         ouverture aléatoire partagée, couleurs échangées
//!                         (arrondi au nombre pair inférieur)
//!   --mentor ""           régime recherche : chemin d'un réseau MENTOR figé.
//!                         L'élève (chess_latest) choisit toujours les coups,
//!                         mais v_racine (étiquettes TD-leaf ET arbitrage)
//!                         vient de la recherche du mentor — anti chambre
//!                         d'écho du TD auto-référentiel (vide = désactivé)
//!   --mentor-poids 1.0    poids de l'étiqueteur EXTERNE (mentor OU oracle)
//!                         dans les étiquettes TD-leaf et l'arbitrage :
//!                         v = poids·v_étiqueteur + (1-poids)·v_élève ;
//!                         1.0 = étiquettes externes pures ; desserrer (0.7)
//!                         laisse la recherche de l'élève ré-entrer dans ses
//!                         étiquettes — sans effet hors --mentor/--oracle
//!   --oracle ""           régime recherche : chemin d'un moteur UCI externe
//!                         (Stockfish) lancé PLEINE FORCE comme étiqueteur —
//!                         même rôle que --mentor (les deux sont EXCLUSIFS),
//!                         mais le plafond de qualité des étiquettes devient
//!                         celui du moteur, plus celui du champion maison ;
//!                         mélange oracle/élève piloté par --mentor-poids,
//!                         comme en mentorat (vide = désactivé)
//!   --oracle-movetime 15  budget (ms) de chaque évaluation de l'oracle
//!   --departs-ouvertures 0  régime recherche : proportion des parties de
//!                         self-play qui démarrent d'une ouverture du livre
//!                         (departs::tirage, rng dérivé de la graine de la
//!                         partie — déterminisme préservé)
//!   --departs-finales 0   idem, proportion des parties qui démarrent d'une
//!                         finale générée
//!   --departs-transition 0  idem, proportion des parties qui démarrent d'un
//!                         MILIEU TARDIF généré (10-16 pièces, matériel
//!                         équilibré, aucune prise gagnante évidente — le
//!                         territoire de conversion fin-de-milieu → finale) ;
//!                         le reste part de la position initiale.
//!                         « 0 0 0 » = comportement historique STRICT
//!                         (aucun tirage, mêmes trajectoires qu'avant).
//!                         Défauts à ZÉRO : options ABSENTES = « 0 0 0 » — une
//!                         ligne de commande historique reste bit-à-bit
//!                         identique à avant ; l'activation des départs variés
//!                         est un opt-in EXPLICITE par flags (valeurs
//!                         conseillées : 0.5, 0.2 et 0.2)
//!   --ancre ""            chemin d'un réseau de RÉFÉRENCE figé : rappel
//!                         élastique DÉCOUPLÉ (style AdamW) appliqué APRÈS
//!                         chaque minibatch (frais ET rejeu) —
//!                         theta -= lr·lambda·(theta - theta_ref). Contient la
//!                         dérive le long des directions plates de l'objectif
//!                         (vide = désactivé, strictement aucun changement)
//!   --ancre-lambda 5.0    intensité du rappel (taux effectif = lr × lambda) ;
//!                         demi-vie d'une dérive non soutenue :
//!                         ln2/(lr·lambda) minibatchs
//!   --recalibrage ""      chemin d'une table FIGÉE label<TAB>v (TSV produit
//!                         par calibration.exe --fit, deux colonnes strictement
//!                         croissantes) : en branche ORACLE, la cible
//!                         d'entraînement vise g(étiquette) au lieu de
//!                         l'étiquette (interpolation linéaire) — le gradient
//!                         cesse de recalibrer l'échelle du réseau et ne
//!                         transporte plus que de l'information d'ordre.
//!                         L'ARBITRAGE reste sur l'étiquette BRUTE (point de
//!                         contrat). Vide = strictement aucun changement
//!   --syzygy ""           dossier des tables de finales Syzygy 3-4-5 (ex.
//!                         engines/syzygy) pour le SELF-PLAY (régime
//!                         recherche) : racine ≤ 5 pièces jouée par DTZ,
//!                         sondes WDL dans l'arbre (src/syzygy.rs) — finales
//!                         parfaites, meilleures étiquettes z. Le mentor
//!                         étiquette avec les mêmes tables. Vide (défaut) =
//!                         strictement aucun changement
//!
//! Régime « recherche » (search_nodes > 0) :
//!   - self-play via selfplay::play_training_game_recherche (un chercheur par
//!     tâche rayon, cibles TD-leaf λ) — moins de positions par cycle que le
//!     régime 1-pli (arbitrage), c'est attendu ;
//!   - mentorat (--mentor non vide) : selfplay::play_training_game_mentor à la
//!     place — DEUX chercheurs par tâche (élève + mentor, mêmes limites de
//!     nœuds), l'élève choisit les coups, la recherche du mentor étiquette ;
//!   - oracle (--oracle non vide, exclusif avec --mentor) : selfplay::
//!     play_training_game_oracle — un POOL de `threads` moteurs UCI pleine
//!     force est lancé au démarrage (Mutex<Vec<UciEngine>>) ; chaque tâche
//!     rayon emprunte un moteur (pop), joue sa partie, le rend (push) —
//!     compatible with_max_len(1), là où map_init relancerait un processus
//!     par partie ; moteur mort ou emprunt raté → relance, et si la relance
//!     échoue la partie se joue SANS oracle (repli élève, jamais de panique) ;
//!   - estimation Elo mesurée avec BotRecherche (1200 nœuds) au lieu de
//!     NetBot d2 : le saut de la courbe Elo au changement de régime est VOULU
//!     (il mesure l'étage recherche) ;
//!   - gating : tous les gate_every cycles, duel BotRecherche(latest) contre
//!     BotRecherche(chess_best.bin) ; promotion (copie latest → best) si
//!     score >= 55 % (promotion directe si best absent ou illisible). Les
//!     deux bots étant déterministes à température 0, chaque paire de parties
//!     tire une ouverture aléatoire commune, jouée des deux couleurs.
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

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Move, Position};

use echec::arena;
use echec::bots::{Bot, BotRecherche, MaterialBot, NetBot, RandomBot};
use echec::checkpoints::{self, TrainState};
use echec::elo;
use echec::features::N_FEATURES;
use echec::nn::{Mlp, SchemaFeatures};
use echec::search;
use echec::selfplay::{self, GameRecord};
use echec::uci::UciEngine;

/// Plis max d'une partie de self-play ou de gating (au-delà : arbitrage en
/// nulle, comme en arène).
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
/// log2 du nombre d'entrées de la TT de chaque chercheur de self-play
/// (2^18 entrées ≈ 262k — un chercheur par tâche rayon, mémoire contenue).
const TAILLE_TT_LOG2_SELFPLAY: u32 = 18;
/// Nœuds par coup des duels de mesure (Elo recherche et gating).
const NOEUDS_DUEL: u64 = 1200;
/// Score minimal du candidat pour être promu champion (gating).
const SEUIL_PROMOTION: f32 = 0.55;
/// Plis d'ouverture aléatoires partagés par chaque paire de parties du gating
/// (les BotRecherche à température 0 étant déterministes, c'est l'ouverture
/// qui diversifie les parties — même schéma que la mini-arène de search.rs).
const PLIS_OUVERTURE_GATING: u32 = 4;

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
    search_nodes: u64,
    td_lambda: f32,
    gate_every: u64,
    gate_games: usize,
    mentor: String,
    mentor_poids: f32,
    oracle: String,
    oracle_movetime: u32,
    departs_ouvertures: f32,
    departs_finales: f32,
    departs_transition: f32,
    ancre: String,
    ancre_lambda: f32,
    recalibrage: String,
    /// Dossier des tables Syzygy 3-4-5 (--syzygy) pour le SELF-PLAY : les
    /// finales de tables y sont jouées parfaitement (racine DTZ, sondes WDL)
    /// → meilleures étiquettes z. Vide (défaut) = comportement historique.
    syzygy: String,
    /// --int8 : la RECHERCHE (self-play, gating, échelle Elo) évalue par le
    /// chemin quantizé de src/quant.rs. L'APPRENTISSAGE reste en f32 —
    /// seule la lecture du réseau par la recherche change. Défaut : false.
    int8: bool,
}

/// Tampon de rejeu DENSE (schéma Classique773) : anneau de positions encodées
/// à capacité fixe.
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

/// Tampon de rejeu CREUX (schéma RoiZones8) : même anneau à capacité fixe que
/// `Rejeu`, mais chaque position est sa liste d'indices actifs (≤ 37 u16 —
/// ~200 fois plus compact que les 773 f32 denses). L'échantillonnage produit
/// directement les lots de `Mlp::train_batch_actifs`.
struct RejeuCreux {
    actifs: Vec<Vec<u16>>,
    zs: Vec<f32>,
    capacite: usize,
    tete: usize,
    len: usize,
}

impl RejeuCreux {
    fn new(capacite: usize) -> Self {
        // Croissance paresseuse, comme `Rejeu`.
        RejeuCreux { actifs: Vec::new(), zs: Vec::new(), capacite, tete: 0, len: 0 }
    }

    fn push(&mut self, a: &[u16], z: f32) {
        if self.len < self.capacite {
            self.actifs.push(a.to_vec());
            self.zs.push(z);
            self.len += 1;
        } else {
            self.actifs[self.tete].clear();
            self.actifs[self.tete].extend_from_slice(a);
            self.zs[self.tete] = z;
        }
        self.tete = (self.tete + 1) % self.capacite;
    }

    /// Minibatch échantillonné uniformément, au format de `train_batch_actifs`.
    fn echantillonne(&self, rng: &mut StdRng, n: usize) -> Vec<(Vec<u16>, f32)> {
        (0..n)
            .map(|_| {
                let i = rng.gen_range(0..self.len);
                (self.actifs[i].clone(), self.zs[i])
            })
            .collect()
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
        search_nodes: 1500,
        td_lambda: 0.3,
        gate_every: 10,
        gate_games: 64,
        mentor: String::new(),
        mentor_poids: 1.0,
        oracle: String::new(),
        oracle_movetime: 15,
        departs_ouvertures: 0.0,
        departs_finales: 0.0,
        departs_transition: 0.0,
        ancre: String::new(),
        ancre_lambda: 5.0,
        recalibrage: String::new(),
        syzygy: String::new(),
        int8: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        // Drapeau SANS valeur : n'avance que d'un cran.
        if nom == "--int8" {
            opt.int8 = true;
            i += 1;
            continue;
        }
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
            "--search-nodes" => opt.search_nodes = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--td-lambda" => opt.td_lambda = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--gate-every" => opt.gate_every = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--gate-games" => opt.gate_games = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--mentor" => opt.mentor = valeur(&args, i, &nom),
            "--mentor-poids" => {
                opt.mentor_poids = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--oracle" => opt.oracle = valeur(&args, i, &nom),
            "--oracle-movetime" => {
                opt.oracle_movetime = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--departs-ouvertures" => {
                opt.departs_ouvertures = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--departs-finales" => {
                opt.departs_finales = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--departs-transition" => {
                opt.departs_transition = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--ancre" => opt.ancre = valeur(&args, i, &nom),
            "--ancre-lambda" => {
                opt.ancre_lambda = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--recalibrage" => opt.recalibrage = valeur(&args, i, &nom),
            "--syzygy" => opt.syzygy = valeur(&args, i, &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                eprintln!(
                    "options : --out --threads --games-per-cycle --temperature --lr \
                     --eval-games --replay --elo-every --elo-games --seed \
                     --search-nodes --td-lambda --gate-every --gate-games --mentor \
                     --mentor-poids --oracle --oracle-movetime \
                     --departs-ouvertures --departs-finales --departs-transition \
                     --ancre --ancre-lambda \
                     --recalibrage --syzygy --int8"
                );
                std::process::exit(2);
            }
        }
        i += 2;
    }
    // Garde-fou des parts de départs : hors [0, 1] ou somme > 1, refus
    // explicite — tirage_complet tronquerait silencieusement les dernières
    // familles et une faute de frappe au déploiement passerait inaperçue.
    if let Err(e) = echec::departs::valide_parts(
        opt.departs_ouvertures,
        opt.departs_finales,
        opt.departs_transition,
    ) {
        eprintln!("{e}");
        std::process::exit(2);
    }
    opt
}

/// Hachage zobrist 64 bits (même convention que selfplay/arena : mode Legal).
fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Copie « atomique » : copie vers `dst`.tmp puis renommage, comme
/// chess_latest.bin. Un lecteur ne voit jamais de fichier partiel — serve.exe
/// recharge à chaud chess_best.bin et les paliers sur changement de mtime, et
/// un arrêt brutal en pleine copie ne doit pas laisser un champion tronqué
/// (il ferait échouer tous les gatings suivants).
fn copie_atomique(src: &str, dst: &str) -> std::io::Result<()> {
    let tmp = format!("{dst}.tmp");
    fs::copy(src, &tmp)?;
    fs::rename(&tmp, dst)
}

/// Append une ligne dans un CSV du dossier modèles (entête à la création).
/// Sert aux journaux lus par le dashboard : gating.csv, events.csv, ancres.csv.
fn append_csv(chemin: &str, entete: &str, ligne: &str) {
    let neuf = !Path::new(chemin).exists();
    if let Ok(mut fichier) = fs::OpenOptions::new().create(true).append(true).open(chemin) {
        if neuf {
            let _ = writeln!(fichier, "{entete}");
        }
        let _ = writeln!(fichier, "{ligne}");
    }
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

/// Limites de duel : `n` nœuds, pas d'autre critère.
fn limites_duel(n: u64) -> search::Limites {
    search::Limites { max_noeuds: n, max_profondeur: 0, movetime_ms: 0 }
}

/// Estimation Elo en régime recherche : duplique elo::mesure avec une fabrique
/// BotRecherche (limites NOEUDS_DUEL nœuds, température 0) — elo.rs reste
/// intact, seul l'agent mesuré change ; même ajuste_elo ensuite. L'échelle
/// d'ancres et le mélange de graines sont IDENTIQUES à elo::mesure pour que
/// seul le bot mesuré diffère entre les deux régimes. Les ancres UCI hautes
/// (stockfish 1700/2000) sont jouées via elo::mesure_uci avec le moteur de
/// `chemin_moteur` (--oracle) ; chemin vide ou moteur injouable → sautées avec
/// message, le fit retombe sur les 5 ancres maison (jamais de panique).
fn mesure_elo_recherche(net: &Arc<Mlp>, parties_par_ancre: usize,
                        graine: u64, chemin_moteur: &str, int8: bool) -> Vec<elo::MesureAncre> {
    let mut mesures: Vec<elo::MesureAncre> = elo::ANCRES
        .iter()
        .filter_map(|a| match a.genre {
            elo::GenreAncre::Maison { profondeur } => Some((a, profondeur)),
            elo::GenreAncre::Uci { .. } => None,
        })
        .enumerate()
        .map(|(k, (a, profondeur))| {
            let net_a = net.clone();
            let score = arena::score(
                move |g: u64| -> Box<dyn Bot> {
                    Box::new(
                        BotRecherche::new(net_a.clone(), g, limites_duel(NOEUDS_DUEL), 0.0)
                            .avec_int8(int8),
                    )
                },
                |g: u64| -> Box<dyn Bot> {
                    match profondeur {
                        None => Box::new(RandomBot::new(g)),
                        Some(d) => Box::new(MaterialBot::new(g, d)),
                    }
                },
                parties_par_ancre,
                graine.wrapping_add(k as u64).wrapping_mul(0x9E37_79B9),
            ) as f64;
            // Progression en direct de la mesure (une ligne par ancre jouée).
            println!("  echelle Elo : {} -> {:.0} % ({} parties)",
                     a.nom, score * 100.0, parties_par_ancre);
            std::io::stdout().flush().ok();
            elo::MesureAncre {
                nom: a.nom,
                elo_ancre: a.elo,
                score,
                parties: parties_par_ancre,
            }
        })
        .collect();
    // Ancres UCI : même agent mesuré (BotRecherche, mêmes limites), moteur
    // bridé en face — mesure_uci imprime sa propre progression et ses sauts.
    let net_u = net.clone();
    mesures.extend(elo::mesure_uci(
        move |g: u64| -> Box<dyn Bot> {
            Box::new(
                BotRecherche::new(net_u.clone(), g, limites_duel(NOEUDS_DUEL), 0.0)
                    .avec_int8(int8),
            )
        },
        chemin_moteur,
        parties_par_ancre,
        graine,
    ));
    mesures
}

/// Une partie du duel de gating : l'ouverture est rejouée depuis la position
/// initiale (ses hachages comptent pour la règle des 3 répétitions), puis
/// candidat et champion (BotRecherche frais, TT vierges, température 0)
/// s'affrontent — le candidat a les blancs si `candidat_blanc`. Mêmes règles
/// de nulle que l'arène : pat, matériel insuffisant, 50 coups, 3e répétition,
/// MAX_PLIES plis (ouverture comprise). Renvoie les points du candidat
/// (1.0 victoire, 0.5 nulle, 0.0 défaite).
fn partie_gating(
    candidat: &Arc<Mlp>,
    champion: &Arc<Mlp>,
    candidat_blanc: bool,
    ouverture: &[Move],
    graine: u64,
    int8: bool,
) -> f32 {
    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    for m in ouverture {
        pos = pos.play(m).expect("coup d'ouverture légal");
        *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
    }
    // Bots frais par partie (TT vierge, équité) ; leurs graines sont inertes à
    // température 0, distinctes par hygiène.
    // Le drapeau int8 s'applique aux DEUX camps : le gating mesure le réseau,
    // pas la voie d'évaluation (mêmes conditions de part et d'autre).
    let mut bot_candidat =
        BotRecherche::new(candidat.clone(), graine, limites_duel(NOEUDS_DUEL), 0.0)
            .avec_int8(int8);
    let mut bot_champion = BotRecherche::new(
        champion.clone(),
        graine.wrapping_add(1),
        limites_duel(NOEUDS_DUEL),
        0.0,
    )
    .avec_int8(int8);
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
        let tour_candidat = (pos.turn() == Color::White) == candidat_blanc;
        let m = if tour_candidat {
            bot_candidat.choose(&pos).expect("coups légaux non vides")
        } else {
            bot_champion.choose(&pos).expect("coups légaux non vides")
        };
        pos = pos.play(&m).expect("coup légal");
        plies += 1;
        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            break 0.0;
        }
    };
    let cote = if candidat_blanc { resultat_blancs } else { -resultat_blancs };
    (cote + 1.0) / 2.0
}

/// Duel de gating : candidat contre champion, tous deux en BotRecherche
/// (NOEUDS_DUEL nœuds, température 0). À budget de nœuds fixe et température
/// nulle, deux BotRecherche sont parfaitement DÉTERMINISTES : sans
/// diversification, toutes les parties d'une même couleur seraient la même
/// trajectoire répétée. Chaque PAIRE de parties tire donc une ouverture
/// aléatoire de PLIS_OUVERTURE_GATING plis, jouée des deux couleurs (équité :
/// chaque camp affronte la même ouverture des deux côtés — le schéma de la
/// mini-arène de search.rs). Paires jouées en parallèle (pool rayon global) ;
/// `parties` est arrondi au nombre pair inférieur (0 ou 1 partie → 0.5).
/// Renvoie le pourcentage de points du candidat dans [0, 1].
fn duel_gating(
    candidat: Arc<Mlp>,
    champion: Arc<Mlp>,
    parties: usize,
    graine: u64,
    int8: bool,
) -> f32 {
    let paires = parties / 2;
    if paires == 0 {
        return 0.5;
    }
    // Progression en direct du duel (une ligne toutes les 4 paires jouées).
    let faites = std::sync::atomic::AtomicUsize::new(0);
    // Une paire par tâche rayon (voir arena::score : sans with_max_len(1),
    // les paquets séquentiels laissent la moitié des ouvriers au chômage).
    let points: f32 = (0..paires)
        .into_par_iter()
        .with_max_len(1)
        .map(|p| {
            // Ouverture aléatoire de la paire. Jamais à court de coups en
            // 4 plis (le mat le plus court en demande 4) ; on s'arrête
            // proprement si une version future allonge l'ouverture.
            let mut rng = StdRng::seed_from_u64(derive_graine(graine, 2 * p as u64));
            let mut pos = Chess::default();
            let mut ouverture: Vec<Move> = Vec::new();
            for _ in 0..PLIS_OUVERTURE_GATING {
                let Some(m) = pos.legal_moves().choose(&mut rng).cloned() else {
                    break;
                };
                pos = pos.play(&m).expect("coup légal");
                ouverture.push(m);
            }
            let g = derive_graine(graine, 2 * p as u64 + 1);
            let pts = partie_gating(&candidat, &champion, true, &ouverture, g, int8)
                + partie_gating(&candidat, &champion, false, &ouverture, g.wrapping_add(2), int8);
            let n = faites.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 4 == 0 || n == paires {
                println!("  gating : {}/{} paires jouees", n, paires);
                std::io::stdout().flush().ok();
            }
            pts
        })
        .sum();
    points / (2 * paires) as f32
}

fn main() {
    echec::pleine_puissance(); // jamais bridé par l'EcoQoS Windows
    let opt = parse_options();

    // Étiqueteurs EXCLUSIFS : le mentor (réseau maison) et l'oracle (moteur
    // UCI externe) remplissent le même rôle — les deux à la fois n'a pas de
    // sens et masquerait une erreur de ligne de commande.
    if !opt.mentor.is_empty() && !opt.oracle.is_empty() {
        eprintln!("--mentor et --oracle sont exclusifs : choisir un seul etiqueteur");
        std::process::exit(2);
    }

    fs::create_dir_all(&opt.out).expect("création du dossier --out");
    // Direct : les parties de self-play retransmettent (une seule à la fois,
    // voir src/direct.rs) dans ce fichier, servi par serve.exe sur /api/live
    // et affiché par la page /live.
    echec::direct::configure(&format!("{}/live.json", opt.out));
    rayon::ThreadPoolBuilder::new()
        .num_threads(opt.threads)
        .build_global()
        .expect("construction du pool rayon global");

    // 1. Reprise : modèle + état cumulés s'ils existent, sinon départ à neuf.
    //    ATTENTION : le départ à neuf crée l'architecture PAR DÉFAUT de
    //    `Mlp::new` ([773,512,64,1]), pas le réseau élargi. Le flux nominal
    //    pour élargir est : distill.exe → copier l'élève sur chess_latest.bin
    //    → relancer train.exe, qui reprend alors les tailles lues du fichier
    //    (tout le chargement est générique). L'architecture est journalisée
    //    ci-dessous pour rendre visible tout départ à neuf inattendu.
    let chemin_latest = checkpoints::latest_path(&opt.out);
    let net = if Path::new(&chemin_latest).exists() {
        Mlp::load(&chemin_latest).expect("chargement de chess_latest.bin")
    } else {
        Mlp::new(opt.seed)
    };
    let mut net = Arc::new(net);
    // Schéma de features SUIVI AUTOMATIQUEMENT du fichier chargé (aucun
    // drapeau) : dense 773 historique ou creux roi-zones — tout le cycle
    // (self-play, apprentissage, rejeu) s'y conforme.
    let schema = net.schema();
    let mut etat = TrainState::load(&opt.out);
    if etat.cycles > 0 {
        println!(
            "reprise : {} cycles, {:.3} h, {} parties, {} positions, architecture {:?}, schema {:?}",
            etat.cycles,
            etat.trained_secs / 3600.0,
            etat.games,
            etat.positions,
            net.sizes,
            schema
        );
    } else {
        println!(
            "réseau neuf (graine {}, architecture {:?}, schema {:?})",
            opt.seed, net.sizes, schema
        );
    }
    // Ancre élastique : réseau de RÉFÉRENCE figé, chargé une fois et jamais
    // modifié. Après CHAQUE minibatch (frais et rejeu), rappel découplé
    // net.rappel_vers(ancre, lr × lambda) — voir Mlp::rappel_vers. Sans
    // --ancre : strictement aucun changement de comportement.
    let ancre: Option<Mlp> = if !opt.ancre.is_empty() {
        match Mlp::load(&opt.ancre) {
            Ok(a) => {
                if a.sizes != net.sizes || a.schema() != schema {
                    eprintln!(
                        "--ancre {} : architecture {:?} / schema {:?} incompatibles \
                         avec le reseau d'entrainement ({:?} / {:?})",
                        opt.ancre,
                        a.sizes,
                        a.schema(),
                        net.sizes,
                        schema
                    );
                    std::process::exit(2);
                }
                println!(
                    "ancre elastique : {} (lambda {}, rappel lr*lambda apres chaque minibatch)",
                    opt.ancre, opt.ancre_lambda
                );
                Some(a)
            }
            Err(e) => {
                eprintln!("--ancre {} : chargement impossible ({e})", opt.ancre);
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    // Mentor (régime recherche uniquement) : réseau FIGÉ chargé une fois au
    // démarrage, dont la recherche étiquette les positions du self-play
    // pendant que l'élève choisit les coups (anti chambre d'écho). Arc
    // DISTINCT de celui du réseau élève : ses clones ne gênent jamais
    // l'Arc::get_mut de l'apprentissage.
    let mentor: Option<Arc<Mlp>> = if opt.search_nodes > 0 && !opt.mentor.is_empty() {
        match Mlp::load(&opt.mentor) {
            Ok(m) => {
                println!(
                    "mentorat : etiquettes par la recherche de {} {:?}",
                    opt.mentor, m.sizes
                );
                Some(Arc::new(m))
            }
            Err(e) => {
                eprintln!("--mentor {} : chargement impossible ({e})", opt.mentor);
                std::process::exit(2);
            }
        }
    } else {
        if !opt.mentor.is_empty() {
            println!("attention : --mentor ignore en regime 1-pli (--search-nodes 0)");
        }
        None
    };
    // Oracle (régime recherche uniquement) : POOL de moteurs UCI PLEINE FORCE
    // lancés UNE FOIS au démarrage, un par thread rayon. Chaque tâche de
    // self-play emprunte un moteur (pop), joue sa partie, le rend (push) —
    // schéma compatible avec with_max_len(1) (une tâche = une partie), là où
    // map_init relancerait un processus par partie. Un échec au lancement est
    // une erreur de configuration (chemin faux, binaire absent) : sortie
    // propre immédiate plutôt qu'une nuit d'étiquettes de repli.
    let oracle_pool: Option<Mutex<Vec<UciEngine>>> =
        if opt.search_nodes > 0 && !opt.oracle.is_empty() {
            let mut moteurs: Vec<UciEngine> = Vec::with_capacity(opt.threads);
            for _ in 0..opt.threads {
                match UciEngine::lance_pleine_force(&opt.oracle, opt.oracle_movetime) {
                    Ok(m) => moteurs.push(m),
                    Err(e) => {
                        eprintln!("--oracle {} : lancement impossible ({e})", opt.oracle);
                        std::process::exit(2);
                    }
                }
            }
            println!("oracle : {} (movetime {} ms)", opt.oracle, opt.oracle_movetime);
            Some(Mutex::new(moteurs))
        } else {
            if !opt.oracle.is_empty() {
                println!("attention : --oracle ignore en regime 1-pli (--search-nodes 0)");
            }
            None
        };
    // Recalibrage des étiquettes oracle : table FIGÉE label → v (TSV de
    // calibration.exe --fit), chargée UNE FOIS au démarrage et jamais
    // modifiée. En branche oracle, la cible d'entraînement vise g(étiquette)
    // au lieu de l'étiquette — l'arbitrage reste sur l'étiquette brute (voir
    // selfplay::Recalibrage). Sans --recalibrage : strictement aucun
    // changement.
    let recalibrage: Option<selfplay::Recalibrage> = if !opt.recalibrage.is_empty() {
        match selfplay::Recalibrage::charge(&opt.recalibrage) {
            Ok(table) => {
                let ((l_min, v_min), (l_max, v_max)) = table.bornes();
                println!(
                    "recalibrage : {} ({} noeuds, label [{l_min:.3}, {l_max:.3}] -> \
                     v [{v_min:.4}, {v_max:.4}]) — cibles oracle seulement, arbitrage brut",
                    opt.recalibrage,
                    table.len()
                );
                if oracle_pool.is_none() {
                    println!(
                        "attention : --recalibrage sans --oracle actif — table chargee \
                         mais sans effet"
                    );
                }
                Some(table)
            }
            Err(e) => {
                eprintln!("--recalibrage {} : {e}", opt.recalibrage);
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    // Tables Syzygy du SELF-PLAY (--syzygy <dossier>, off par défaut) :
    // chargées UNE FOIS, partagées par Arc entre tous les chercheurs des
    // tâches rayon (les tables sont Sync, sondables en parallèle). Les
    // finales ≤ 5 pièces sont alors jouées PARFAITEMENT (racine DTZ) et les
    // sous-arbres de tables reçoivent des verdicts exacts (sondes WDL) :
    // meilleures étiquettes z. Une erreur de chargement est une erreur de
    // configuration : sortie propre immédiate.
    let syzygy: Option<Arc<echec::syzygy::Tables>> = if !opt.syzygy.is_empty() {
        match echec::syzygy::charge(&opt.syzygy) {
            Ok((tables, n)) => {
                println!("syzygy : {n} tables chargées depuis {}", opt.syzygy);
                Some(Arc::new(tables))
            }
            Err(e) => {
                eprintln!("--syzygy {} : {e}", opt.syzygy);
                std::process::exit(2);
            }
        }
    } else {
        None
    };
    // Départs variés (régime recherche uniquement) : proportion des parties
    // qui démarrent d'une ouverture du livre / d'une finale générée / d'un
    // milieu tardif généré. « 0 0 0 » = comportement historique STRICT
    // (aucun tirage).
    let utilise_departs = opt.search_nodes > 0
        && (opt.departs_ouvertures > 0.0
            || opt.departs_finales > 0.0
            || opt.departs_transition > 0.0);
    if opt.search_nodes > 0 {
        println!(
            "departs : ouvertures {:.0} %, finales {:.0} %, transition {:.0} %",
            opt.departs_ouvertures * 100.0,
            opt.departs_finales * 100.0,
            opt.departs_transition * 100.0
        );
    }
    // Marqueur de changement de régime pour les courbes du dashboard : posé une
    // seule fois, à la première activation du régime recherche.
    if opt.search_nodes > 0 {
        let chemin_events = format!("{}/events.csv", opt.out);
        let deja = fs::read_to_string(&chemin_events)
            .map(|s| s.contains("recherche"))
            .unwrap_or(false);
        if !deja {
            append_csv(
                &chemin_events,
                "elapsed_hours,label",
                &format!("{:.3},recherche", etat.trained_secs / 3600.0),
            );
        }
    }
    // Marqueur « oracle » : même mécanisme, posé une seule fois à la première
    // activation de l'étiquetage par moteur externe.
    if oracle_pool.is_some() {
        let chemin_events = format!("{}/events.csv", opt.out);
        let deja = fs::read_to_string(&chemin_events)
            .map(|s| s.contains("oracle"))
            .unwrap_or(false);
        if !deja {
            append_csv(
                &chemin_events,
                "elapsed_hours,label",
                &format!("{:.3},oracle", etat.trained_secs / 3600.0),
            );
        }
    }

    // Tampon de rejeu au format du schéma : dense (f32 concaténés) pour
    // Classique773, creux (listes d'indices) pour RoiZones8.
    let mut rejeu = (opt.replay > 0 && schema == SchemaFeatures::Classique773)
        .then(|| Rejeu::new(opt.replay));
    let mut rejeu_creux = (opt.replay > 0 && schema == SchemaFeatures::RoiZones8)
        .then(|| RejeuCreux::new(opt.replay));
    if let Some(r) = &rejeu {
        println!(
            "tampon de rejeu : {} positions max (~{:.1} Go)",
            r.capacite,
            (r.capacite * N_FEATURES * 4) as f64 / 1e9
        );
    }
    if let Some(r) = &rejeu_creux {
        // ~37 u16 + l'entête du Vec par position : ~100 octets.
        println!(
            "tampon de rejeu (creux) : {} positions max (~{:.2} Go)",
            r.capacite,
            (r.capacite * 100) as f64 / 1e9
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
        // Direct : numéro du cycle affiché par la page /live — etat.cycles est
        // incrémenté en FIN de cycle, celui qui se joue est donc le suivant.
        echec::direct::annonce_cycle(etat.cycles + 1);

        // 2. Self-play : graines dérivées de seed + parties déjà jouées, pour
        //    qu'une reprise continue exactement la séquence de parties.
        let graines: Vec<u64> = (0..opt.games_per_cycle)
            .map(|i| opt.seed.wrapping_add(etat.games).wrapping_add(i as u64))
            .collect();
        let parties: Vec<GameRecord> = if opt.search_nodes > 0 {
            // Régime recherche : chaque tâche rayon crée SON chercheur (TT
            // locale, clone d'Arc du réseau) et joue une partie TD-leaf —
            // en mentorat, DEUX chercheurs (élève + mentor, même taille de
            // TT), les coups à l'élève, les étiquettes au mentor ; en mode
            // oracle, un moteur UCI emprunté au pool étiquette à la place.
            // Les Recherche — donc les clones d'Arc du réseau ÉLÈVE — sont
            // créés et droppés À L'INTÉRIEUR de chaque fermeture map : à la
            // sortie du collect, seul l'Arc principal survit et
            // l'Arc::get_mut de l'apprentissage réussit (les Arc du mentor
            // pointent un réseau distinct et ne le gênent pas).
            // NB : --temperature ne s'applique PAS ici (régime 1-pli
            // uniquement, voir l'en-tête) — les températures du régime
            // recherche (0.2, ouverture 0.8) sont les défauts FIGÉS du
            // contrat OptionsRecherche, repris par ..Default::default().
            let opts_recherche = selfplay::OptionsRecherche {
                nodes_par_coup: opt.search_nodes,
                lambda: opt.td_lambda,
                max_plies: MAX_PLIES,
                poids_prof: opt.mentor_poids,
                ..Default::default()
            };
            // Progression en direct : une ligne toutes les 8 parties terminées,
            // pour que `Get-Content train.log -Wait` montre le calcul en cours
            // et pas seulement les fins de cycle.
            let fait = std::sync::atomic::AtomicUsize::new(0);
            let total = graines.len();
            graines
                .par_iter()
                .with_max_len(1)
                .map(|&g| {
                    let mut eleve =
                        search::Recherche::new(net.clone(), TAILLE_TT_LOG2_SELFPLAY);
                    // --int8 : la recherche de self-play évalue en quantizé
                    // (les étiquettes TD-leaf restent des scores [-1,1]).
                    eleve.utilise_int8 = opt.int8;
                    // --syzygy : finales de tables jouées parfaitement
                    // (clone d'Arc, tables partagées entre les tâches).
                    eleve.syzygy = syzygy.clone();
                    // Départ de la partie : ouverture du livre / finale
                    // générée / milieu tardif généré (transition) / position
                    // initiale, tiré d'un rng DÉRIVÉ de la
                    // graine de la partie (déterminisme : même graine → même
                    // départ, reprise comprise). None = variantes historiques,
                    // trajectoires strictement identiques à avant.
                    let depart = utilise_departs.then(|| {
                        let mut rng_depart = StdRng::seed_from_u64(derive_graine(g, 0xDE9A47));
                        // tirage_complet : part de transition à 0 = tirage
                        // historique à l'identique (même consommation du rng).
                        echec::departs::tirage_complet(
                            &mut rng_depart,
                            opt.departs_ouvertures,
                            opt.departs_finales,
                            opt.departs_transition,
                        )
                    });
                    let partie = if let Some(pool) = &oracle_pool {
                        // Emprunt d'un moteur au pool : pop sous verrou, le
                        // verrou est relâché PENDANT la partie (le guard est
                        // un temporaire de l'instruction). Un verrou
                        // empoisonné est récupéré via into_inner — jamais de
                        // panique dans une tâche de self-play.
                        let emprunte = pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
                        // Santé du moteur emprunté : un isready draine les
                        // restes éventuels et détecte un processus mort.
                        let vivant = match emprunte {
                            Some(mut m) => m.pret().is_ok().then_some(m),
                            None => None,
                        };
                        // Moteur mort ou emprunt raté (pool vide) → relance ;
                        // relance impossible → la partie se joue SANS oracle
                        // (repli élève, auto-étiquetée).
                        let mut moteur = vivant.or_else(|| {
                            UciEngine::lance_pleine_force(&opt.oracle, opt.oracle_movetime)
                                .ok()
                        });
                        let partie = match (moteur.as_mut(), &depart) {
                            (Some(o), Some(d)) => selfplay::play_training_game_oracle_depuis(
                                &mut eleve,
                                o,
                                d,
                                g,
                                &opts_recherche,
                                recalibrage.as_ref(),
                            ),
                            (Some(o), None) => selfplay::play_training_game_oracle(
                                &mut eleve,
                                o,
                                g,
                                &opts_recherche,
                                recalibrage.as_ref(),
                            ),
                            (None, Some(d)) => selfplay::play_training_game_recherche_depuis(
                                &mut eleve,
                                d,
                                g,
                                &opts_recherche,
                            ),
                            (None, None) => selfplay::play_training_game_recherche(
                                &mut eleve,
                                g,
                                &opts_recherche,
                            ),
                        };
                        // Restitution (un moteur relancé remplace le mort :
                        // la taille du pool reste stable).
                        if let Some(m) = moteur {
                            pool.lock().unwrap_or_else(|e| e.into_inner()).push(m);
                        }
                        partie
                    } else if let Some(m) = &mentor {
                        let mut prof =
                            search::Recherche::new(m.clone(), TAILLE_TT_LOG2_SELFPLAY);
                        prof.utilise_int8 = opt.int8;
                        // Le mentor étiquette avec les mêmes tables : ses
                        // verdicts de finale sont exacts eux aussi.
                        prof.syzygy = syzygy.clone();
                        match &depart {
                            Some(d) => selfplay::play_training_game_mentor_depuis(
                                &mut eleve,
                                &mut prof,
                                d,
                                g,
                                &opts_recherche,
                            ),
                            None => selfplay::play_training_game_mentor(
                                &mut eleve,
                                &mut prof,
                                g,
                                &opts_recherche,
                            ),
                        }
                    } else {
                        match &depart {
                            Some(d) => selfplay::play_training_game_recherche_depuis(
                                &mut eleve,
                                d,
                                g,
                                &opts_recherche,
                            ),
                            None => selfplay::play_training_game_recherche(
                                &mut eleve,
                                g,
                                &opts_recherche,
                            ),
                        }
                    };
                    let n = fait.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                    if n % 8 == 0 || n == total {
                        println!("  self-play : {n}/{total} parties");
                        std::io::stdout().flush().ok();
                    }
                    partie
                })
                .collect()
        } else {
            // Ancien régime 1-pli, intact.
            let net_ref: &Mlp = &net;
            graines
                .par_iter()
                .with_max_len(1)
                .map(|&g| selfplay::play_training_game(net_ref, g, opt.temperature, MAX_PLIES))
                .collect()
        };

        // Concatène toutes les positions du cycle : xs dense (Classique773)
        // OU listes d'indices actifs (RoiZones8), selon le schéma du réseau —
        // le self-play a rempli le bon champ de chaque GameRecord.
        let n_positions: usize = parties.iter().map(|p| p.zs.len()).sum();
        let mut xs: Vec<f32> = Vec::new();
        let mut actifs: Vec<Vec<u16>> = Vec::new();
        let mut zs: Vec<f32> = Vec::with_capacity(n_positions);
        if schema == SchemaFeatures::Classique773 {
            xs.reserve(n_positions * N_FEATURES);
        } else {
            actifs.reserve(n_positions);
        }
        for p in parties {
            xs.extend(p.xs);
            actifs.extend(p.actifs);
            zs.extend(p.zs);
        }

        // 3. Apprentissage : indices mélangés, minibatchs de 256, 1 époque,
        //    dernier lot partiel accepté. Loss = moyenne pondérée par la taille
        //    des lots. (Aucun Arc cloné à ce stade : get_mut réussit.)
        //    Chemin DENSE (train_batch) ou CREUX (train_batch_actifs) selon le
        //    schéma — mêmes minibatchs, même optimiseur, même comptage de loss.
        let net_mut = Arc::get_mut(&mut net).expect("réseau encore partagé à l'apprentissage");
        let mut indices: Vec<usize> = (0..n_positions).collect();
        let mut rng_melange =
            StdRng::seed_from_u64(derive_graine(opt.seed.wrapping_add(etat.cycles), 0x5AFF1E));
        indices.shuffle(&mut rng_melange);
        let mut somme_loss = 0.0f64;
        let mut n_vus = 0usize;
        match schema {
            SchemaFeatures::Classique773 => {
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
                    // Rappel élastique découplé, APRÈS le pas Adam.
                    if let Some(a) = &ancre {
                        net_mut.rappel_vers(a, opt.lr * opt.ancre_lambda);
                    }
                    somme_loss += loss_lot as f64 * lot.len() as f64;
                    n_vus += lot.len();
                }

                // 3 bis. Rejeu : les positions fraîches entrent dans le tampon,
                // puis on rejoue autant de minibatchs, tirés uniformément dans
                // TOUT le tampon (frais + anciens). Chaque position finit donc
                // revue plusieurs fois au fil des cycles avant d'être écrasée.
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
                            if let Some(a) = &ancre {
                                net_mut.rappel_vers(a, opt.lr * opt.ancre_lambda);
                            }
                            somme_loss += loss_lot as f64 * MINIBATCH as f64;
                            n_vus += MINIBATCH;
                        }
                    }
                }
            }
            SchemaFeatures::RoiZones8 => {
                for lot in indices.chunks(MINIBATCH) {
                    let lots: Vec<(Vec<u16>, f32)> = lot
                        .iter()
                        .map(|&i| (actifs[i].clone(), zs[i]))
                        .collect();
                    let loss_lot = net_mut.train_batch_actifs(&lots, opt.lr);
                    // Rappel élastique découplé, APRÈS le pas Adam.
                    if let Some(a) = &ancre {
                        net_mut.rappel_vers(a, opt.lr * opt.ancre_lambda);
                    }
                    somme_loss += loss_lot as f64 * lot.len() as f64;
                    n_vus += lot.len();
                }

                // 3 bis. Rejeu creux : même politique que le rejeu dense.
                if let Some(r) = rejeu_creux.as_mut() {
                    for i in 0..n_positions {
                        r.push(&actifs[i], zs[i]);
                    }
                    if r.len >= MINIBATCH {
                        let mut rng_rejeu = StdRng::seed_from_u64(derive_graine(
                            opt.seed.wrapping_add(etat.cycles),
                            0x8E3E0,
                        ));
                        for _ in 0..n_positions.div_ceil(MINIBATCH) {
                            let lots = r.echantillonne(&mut rng_rejeu, MINIBATCH);
                            let loss_lot = net_mut.train_batch_actifs(&lots, opt.lr);
                            if let Some(a) = &ancre {
                                net_mut.rappel_vers(a, opt.lr * opt.ancre_lambda);
                            }
                            somme_loss += loss_lot as f64 * MINIBATCH as f64;
                            n_vus += MINIBATCH;
                        }
                    }
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
        // Une mise en veille au MILIEU d'un cycle gonflerait trained_secs
        // d'heures fantômes (le chrono mural tourne pendant le sommeil) : la
        // durée créditée est plafonnée à 15 min — aucun cycle réel n'approche
        // ce plafond (médiane ~3 min). Leçon des 16,9 h fantômes purgées des
        // courbes le 28/07.
        let duree_brute = debut_cycle.elapsed().as_secs_f64();
        let duree_cycle = if duree_brute > 900.0 {
            println!(
                "  (cycle de {duree_brute:.0} s plafonné à 900 s — veille pendant le cycle ?)"
            );
            900.0
        } else {
            duree_brute
        };
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
            copie_atomique(&chemin_latest, &chemin_palier)
                .expect("copie de l'instantané de palier");
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
            // En régime recherche, l'agent mesuré est BotRecherche (1200 nœuds),
            // comme ce que le gating et le serveur font jouer : le saut de la
            // courbe au changement de régime est voulu (il mesure l'étage
            // recherche). Sinon, mesure historique NetBot d2.
            let graine_elo = derive_graine(opt.seed.wrapping_add(etat.cycles), 0xE10);
            let mesures = if opt.search_nodes > 0 {
                // Le moteur des ancres UCI est celui de --oracle (réutilisé) ;
                // absent → mesure_elo_recherche saute proprement ces ancres.
                mesure_elo_recherche(&net, opt.elo_games, graine_elo, &opt.oracle, opt.int8)
            } else {
                elo::mesure(&net, PROFONDEUR_ELO, opt.elo_games, graine_elo)
            };
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
            // Journal par ancre lu par la page /training (courbe des ancres) :
            // une ligne par ancre effectivement jouée à cette mesure.
            for m in &mesures {
                append_csv(
                    &format!("{}/ancres.csv", opt.out),
                    "heures,ancre,score_pct,parties",
                    &format!("{:.3},{},{:.1},{}", apres_h, m.nom, m.score * 100.0, m.parties),
                );
            }
            let detail: Vec<String> = mesures
                .iter()
                .map(|m| format!("{} {:.0} %", m.nom, m.score * 100.0))
                .collect();
            println!(
                "  Elo estime ~{:.0} (echelle d'ancres ; {})",
                estimation,
                detail.join(", ")
            );
            std::io::stdout().flush().ok();
        }

        // 9. Gating (régime recherche uniquement) : le dernier réseau doit
        //    détrôner le champion chess_best.bin en duel BotRecherche contre
        //    BotRecherche pour être promu. Hors chronométrage des paliers,
        //    comme l'Elo.
        if opt.search_nodes > 0 && opt.gate_every > 0 && etat.cycles % opt.gate_every == 0 {
            let chemin_best = format!("{}/chess_best.bin", opt.out);
            if !Path::new(&chemin_best).exists() {
                // Pas encore de champion : promotion directe.
                copie_atomique(&chemin_latest, &chemin_best).expect("copie latest -> best");
                println!("gating : promu (pas de champion, promotion directe)");
            } else {
                match Mlp::load(&chemin_best) {
                    // Champion illisible (ex. fichier tronqué hérité d'un
                    // arrêt brutal) : promotion directe de secours plutôt
                    // qu'un panic qui tuerait la nuit d'entraînement à
                    // chaque cycle multiple de gate_every.
                    Err(e) => {
                        copie_atomique(&chemin_latest, &chemin_best)
                            .expect("copie latest -> best (secours)");
                        println!(
                            "gating : champion illisible ({e}) -> promotion directe de latest"
                        );
                    }
                    Ok(champion) => {
                        let score = duel_gating(
                            net.clone(),
                            Arc::new(champion),
                            opt.gate_games,
                            derive_graine(opt.seed.wrapping_add(etat.cycles), 0x6A7E),
                            opt.int8,
                        );
                        let promu = score >= SEUIL_PROMOTION;
                        if promu {
                            copie_atomique(&chemin_latest, &chemin_best)
                                .expect("copie latest -> best");
                            println!("gating : promu ({:.0} %)", score * 100.0);
                        } else {
                            println!("gating : refuse ({:.0} %)", score * 100.0);
                        }
                        // Journal lu par la page /training.
                        append_csv(
                            &format!("{}/gating.csv", opt.out),
                            "elapsed_hours,score_pct,promu",
                            &format!("{:.3},{:.1},{}", apres_h, score * 100.0,
                                     if promu { 1 } else { 0 }),
                        );
                    }
                }
            }
            std::io::stdout().flush().ok();
        }
    }
}
