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
//!
//! Régime « recherche » (search_nodes > 0) :
//!   - self-play via selfplay::play_training_game_recherche (un chercheur par
//!     tâche rayon, cibles TD-leaf λ) — moins de positions par cycle que le
//!     régime 1-pli (arbitrage), c'est attendu ;
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
use std::sync::Arc;
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
use echec::nn::Mlp;
use echec::search;
use echec::selfplay::{self, GameRecord};

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
        search_nodes: 1500,
        td_lambda: 0.3,
        gate_every: 10,
        gate_games: 64,
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
            "--search-nodes" => opt.search_nodes = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--td-lambda" => opt.td_lambda = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--gate-every" => opt.gate_every = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--gate-games" => opt.gate_games = parse_valeur(&valeur(&args, i, &nom), &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                eprintln!(
                    "options : --out --threads --games-per-cycle --temperature --lr \
                     --eval-games --replay --elo-every --elo-games --seed \
                     --search-nodes --td-lambda --gate-every --gate-games"
                );
                std::process::exit(2);
            }
        }
        i += 2;
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
/// Sert aux journaux lus par le dashboard : gating.csv, events.csv.
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
/// seul le bot mesuré diffère entre les deux régimes.
fn mesure_elo_recherche(net: &Arc<Mlp>, parties_par_ancre: usize,
                        graine: u64) -> Vec<elo::MesureAncre> {
    elo::ANCRES
        .iter()
        .enumerate()
        .map(|(k, a)| {
            let net_a = net.clone();
            let score = arena::score(
                move |g: u64| -> Box<dyn Bot> {
                    Box::new(BotRecherche::new(
                        net_a.clone(),
                        g,
                        limites_duel(NOEUDS_DUEL),
                        0.0,
                    ))
                },
                |g: u64| -> Box<dyn Bot> {
                    match a.profondeur {
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
        .collect()
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
    let mut bot_candidat =
        BotRecherche::new(candidat.clone(), graine, limites_duel(NOEUDS_DUEL), 0.0);
    let mut bot_champion = BotRecherche::new(
        champion.clone(),
        graine.wrapping_add(1),
        limites_duel(NOEUDS_DUEL),
        0.0,
    );
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
fn duel_gating(candidat: Arc<Mlp>, champion: Arc<Mlp>, parties: usize, graine: u64) -> f32 {
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
            let pts = partie_gating(&candidat, &champion, true, &ouverture, g)
                + partie_gating(&candidat, &champion, false, &ouverture, g.wrapping_add(2));
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
        let parties: Vec<GameRecord> = if opt.search_nodes > 0 {
            // Régime recherche : chaque tâche rayon crée SON chercheur (TT
            // locale, clone d'Arc du réseau) et joue une partie TD-leaf.
            // Les Recherche — donc les clones d'Arc — sont créés et droppés
            // À L'INTÉRIEUR de chaque fermeture map : à la sortie du collect,
            // seul l'Arc principal survit et l'Arc::get_mut de l'apprentissage
            // réussit.
            // NB : --temperature ne s'applique PAS ici (régime 1-pli
            // uniquement, voir l'en-tête) — les températures du régime
            // recherche (0.2, ouverture 0.8) sont les défauts FIGÉS du
            // contrat OptionsRecherche, repris par ..Default::default().
            let opts_recherche = selfplay::OptionsRecherche {
                nodes_par_coup: opt.search_nodes,
                lambda: opt.td_lambda,
                max_plies: MAX_PLIES,
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
                    let mut recherche =
                        search::Recherche::new(net.clone(), TAILLE_TT_LOG2_SELFPLAY);
                    let partie =
                        selfplay::play_training_game_recherche(&mut recherche, g, &opts_recherche);
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
                mesure_elo_recherche(&net, opt.elo_games, graine_elo)
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
