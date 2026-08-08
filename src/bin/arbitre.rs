//! L'ARBITRE : annote en direct chaque coup du match, des DEUX camps.
//!
//! Un processus minuscule qui SURVEILLE models/match_live.json (écrit
//! atomiquement par match.exe à chaque coup) et, pour chaque demi-coup joué,
//! demande à un moteur UCI ce qu'il aurait joué, puis mesure ce que le coup
//! réellement joué a coûté. Sorties : models/match_arbitre.json (servi sur
//! GET /api/arbitre, affiché par le panneau « arbitre » de la page /match) et
//! models/arbitre.csv (une ligne par pli, pour l'analyse ultérieure).
//!
//! Il ne touche à RIEN de ce qui tourne : lecture seule de match_live.json,
//! aucun accès au réseau ni à la recherche maison, son propre processus
//! Stockfish bridé à 1 thread.
//!
//! ── ÉCONOMIE D'ANALYSES (le point de conception) ───────────────────────────
//! Annoter un coup demande DEUX évaluations : la position avant et la position
//! après. Les demander séparément coûterait 2 analyses par pli. Or la position
//! APRÈS le pli i est exactement la position AVANT le pli i+1 : une seule
//! analyse par POSITION suffit, et chacune sert deux fois (« après » du coup
//! précédent, « avant » du coup suivant). Un cache indexé par FEN matérialise
//! ce partage — soit N+1 analyses pour N plis au lieu de 2N, la moitié du
//! temps moteur. Aucune latence en échange : la position après le dernier coup
//! joué est déjà publiée dans history_fen, l'annotation du dernier pli n'attend
//! donc pas le coup suivant.
//!
//! ── REPRISE ────────────────────────────────────────────────────────────────
//! models/arbitre.csv est la mémoire : au démarrage il est relu, et l'arbitre
//! repart au premier pli non consigné de la partie en cours (echec::arbitre::
//! prochain_pli). Relancer l'arbitre ne réanalyse donc rien — sauf si le CSV
//! ne correspond PLUS au match diffusé (nouveau match reparti à la partie 1,
//! coups différents pour un même pli) : cette incohérence est détectée en
//! comparant les coups joués consignés à history_san, et le CSV est alors
//! ARCHIVÉ (renommé, jamais effacé) pour repartir sur une base saine. Un
//! verrou <csv>.lock interdit deux arbitres sur le même CSV (double coût
//! moteur et résumé faussé).
//!
//! ── FIN DE PARTIE ──────────────────────────────────────────────────────────
//! match.rs n'insère aucune pause entre deux parties : la publication qui porte
//! le résultat est suivie, en une poignée de secondes, du premier état de la
//! partie suivante avec un historique VIDE. Comme l'arbitre met plusieurs
//! secondes par pli, les derniers coups — le mat, la gaffe décisive — seraient
//! perdus. Au changement de partie, il termine donc la partie close avant de
//! tourner la page, en s'appuyant sur le dernier état vu ET sur le PGN écrit
//! par match.rs (--pgn, la source complète même si le dernier état du direct
//! n'a jamais été échantillonné).
//!
//! ── RÉSERVE SUR LE MÈTRE-ÉTALON ────────────────────────────────────────────
//! L'arbitre est le MÊME moteur que le Fantôme, en pleine force. Les coups du
//! Fantôme sortent donc de la fonction d'évaluation qui les note : sa perte
//! moyenne est mécaniquement tirée vers le bas par rapport à un moteur d'une
//! autre famille de force égale. La ligne « Fantôme » du panneau situe le
//! champion, elle ne le mesure pas : l'écart champion/Fantôme est structurel-
//! lement surestimé. (Le bruit d'analyse, lui, est faible : sur les plis où le
//! coup joué EST celui du moteur, la perte mesurée vaut ~1 cp à movetime 1000.)
//!
//! ── COÛT CPU ───────────────────────────────────────────────────────────────
//! La machine tourne déjà à plein (match SMP + ponder). L'arbitre s'impose donc
//! : 1 thread de moteur (--threads), un hash modeste (--hash), un movetime
//! court (--movetime), une pause entre deux analyses (--pause) et, une fois à
//! jour, un simple sondage du fichier toutes les 2 s — jamais de rafale.
//!
//! Exemple (fume-test sur le match EN COURS, sans le gêner) :
//!   arbitre --movetime 1000 --threads 1 --hash 128 --pause 500

use std::collections::{HashMap, HashSet};
use std::io::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use shakmaty::fen::Fen;
use shakmaty::san::SanPlus;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Position};

use echec::arbitre::{
    camp, classement, coups_du_pgn, cp_du_score, csv_coherent, fens_de_coups, lit_csv, perte_cp,
    phase, prochain_pli_hors, resume, vers_blancs, Annotation, ENTETE_CSV, MAT_CP,
};
use echec::uci::UciEngine;

/// Sondage du fichier du direct quand il n'y a rien à analyser (à jour, ou
/// match_live.json absent/illisible).
const ATTENTE_MS: u64 = 2_000;
/// Tentatives d'analyse d'une même position avant d'abandonner ce tour de
/// boucle (le moteur est relancé entre deux).
const ESSAIS_MOTEUR: u32 = 3;
/// Tours de boucle ratés sur un MÊME pli avant de l'abandonner et de passer au
/// suivant. Sans ce garde-fou, une position que le moteur refuse d'évaluer
/// (FEN illisible, bestmove sans score) bloquerait l'arbitre pour toujours sur
/// ce pli — et, dans le second cas, en consommant ESSAIS_MOTEUR × movetime par
/// tour, soit une rafale continue que la machine ne peut pas se permettre.
const ESSAIS_PLI: u32 = 3;

/// Arrêt demandé (Ctrl-C) : la boucle sort à la première occasion, après
/// l'analyse en cours — les fichiers sont donc toujours cohérents.
static ARRET: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Opt {
    live: String,
    engine: String,
    movetime: u64,
    threads: u32,
    hash: u32,
    out: String,
    csv: String,
    pause: u64,
    /// Dossier des PGN du match — le même que le `--pgn` de match.exe. Sert au
    /// seul rattrapage de fin de partie ; absent ou vide, l'arbitre s'en passe.
    pgn: String,
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
        live: "models/match_live.json".to_string(),
        engine: "engines/stockfish/stockfish-windows-x86-64-avx2.exe".to_string(),
        movetime: 4_000,
        threads: 1,
        hash: 256,
        out: "models/match_arbitre.json".to_string(),
        csv: "models/arbitre.csv".to_string(),
        pause: 250,
        pgn: "pgn_match_2600".to_string(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--live" => opt.live = valeur(&args, i, &nom),
            "--engine" => opt.engine = valeur(&args, i, &nom),
            "--movetime" => opt.movetime = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--threads" => opt.threads = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--hash" => opt.hash = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--out" => opt.out = valeur(&args, i, &nom),
            "--csv" => opt.csv = valeur(&args, i, &nom),
            "--pause" => opt.pause = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--pgn" => opt.pgn = valeur(&args, i, &nom),
            _ => {
                eprintln!("option inconnue : {nom}");
                eprintln!(
                    "usage : arbitre [--live models/match_live.json] [--engine <exe>] \
                     [--movetime 4000] [--threads 1] [--hash 256] \
                     [--out models/match_arbitre.json] [--csv models/arbitre.csv] \
                     [--pause 250] [--pgn pgn_match_2600]"
                );
                std::process::exit(2);
            }
        }
        i += 2;
    }
    // NON-PERTURBATION DU DIRECT : l'arbitre ne doit JAMAIS écrire dans
    // models/match_live.json — une faute de frappe sur --out (ou pire, --csv,
    // que l'archivage RENOMMERAIT, faisant disparaître le match de la page)
    // suffirait. Le contrôle est fait ici, avant toute ouverture de fichier.
    for (drapeau, chemin) in [("--out", &opt.out), ("--csv", &opt.csv)] {
        if meme_fichier(chemin, &opt.live) {
            eprintln!(
                "option {drapeau} : {chemin} désigne le fichier du direct ({}) — refusé, \
                 l'arbitre n'écrit jamais dedans.",
                opt.live
            );
            std::process::exit(2);
        }
    }
    if meme_fichier(&opt.out, &opt.csv) {
        eprintln!("options --out et --csv : même fichier ({}) — refusé.", opt.out);
        std::process::exit(2);
    }
    opt
}

/// Deux chemins désignent-ils le même fichier ? Comparaison canonique quand
/// les deux existent (« models/x » et « ./models/../models/x »), sinon
/// comparaison littérale — un chemin de sortie qui n'existe pas encore ne peut
/// de toute façon pas être le fichier du direct, qui, lui, existe.
fn meme_fichier(a: &str, b: &str) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

// ---------------------------------------------------------------------------
// Ctrl-C
// ---------------------------------------------------------------------------

/// Ctrl-C (et fermeture de console) : le gestionnaire pose seulement le
/// drapeau ARRET et rend TRUE — le processus N'EST PAS tué sur place, la
/// boucle principale sort d'elle-même et referme proprement moteur, CSV et
/// JSON. Sans dépendance : kernel32 est toujours lié (même approche que la
/// garde mémoire de src/bin/match.rs).
#[cfg(windows)]
fn arme_ctrl_c() {
    extern "system" fn gestionnaire(_type_evenement: u32) -> i32 {
        ARRET.store(true, Ordering::SeqCst);
        1 // TRUE : signal traité
    }
    extern "system" {
        fn SetConsoleCtrlHandler(
            gestionnaire: Option<extern "system" fn(u32) -> i32>,
            ajout: i32,
        ) -> i32;
    }
    // SÛRETÉ : pointeur de fonction 'static, la convention system est celle
    // attendue par l'API ; aucun tampon partagé.
    unsafe {
        SetConsoleCtrlHandler(Some(gestionnaire), 1);
    }
}

#[cfg(not(windows))]
fn arme_ctrl_c() {}

fn arret_demande() -> bool {
    ARRET.load(Ordering::SeqCst)
}

/// Sommeil interruptible : découpé en tranches de 100 ms pour que Ctrl-C
/// n'attende jamais la fin d'une longue attente.
fn dors(ms: u64) {
    let mut reste = ms;
    while reste > 0 && !arret_demande() {
        let tranche = reste.min(100);
        std::thread::sleep(Duration::from_millis(tranche));
        reste -= tranche;
    }
}

// ---------------------------------------------------------------------------
// Lecture du direct (lecture SEULE : rien n'est jamais écrit dans ce fichier)
// ---------------------------------------------------------------------------

/// Clone : le dernier état vu est CONSERVÉ d'un tour à l'autre pour servir de
/// source de rattrapage quand la partie change (voir `rattrape_fin_de_partie`).
#[derive(Clone)]
struct Live {
    partie: u32,
    champion_blanc: bool,
    termine: bool,
    /// Coups joués, en SAN (history_san[i] = coup du pli i+1).
    san: Vec<String>,
    /// FEN après chaque pli, position initiale incluse : fen[i] = position
    /// AVANT le coup du pli i+1.
    fen: Vec<String>,
}

impl Live {
    /// Nombre de plis exploitables : le contrat garantit
    /// fen.len() == san.len() + 1 ; on prend l'intersection pour ne jamais
    /// indexer hors bornes si le contrat évoluait.
    fn plis(&self) -> usize {
        self.san.len().min(self.fen.len().saturating_sub(1))
    }
}

/// Lit models/match_live.json. None si le fichier est absent, momentanément
/// illisible (il est pourtant écrit atomiquement : ceinture et bretelles), ou
/// d'une génération sans history_fen — dans tous les cas l'arbitre attend, il
/// n'échoue pas.
fn lit_live(chemin: &str) -> Option<Live> {
    let texte = std::fs::read_to_string(chemin).ok()?;
    let v: serde_json::Value = serde_json::from_str(&texte).ok()?;
    let tableau = |nom: &str| -> Vec<String> {
        v.get(nom)
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    let fen = tableau("history_fen");
    if fen.is_empty() {
        return None;
    }
    Some(Live {
        partie: v.get("partie").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        champion_blanc: v
            .get("champion_blanc")
            .and_then(|x| x.as_bool())
            .unwrap_or(true),
        termine: v.get("termine").and_then(|x| x.as_bool()).unwrap_or(false),
        san: tableau("history_san"),
        fen,
    })
}

// ---------------------------------------------------------------------------
// Analyse d'une position
// ---------------------------------------------------------------------------

/// Verdict du moteur sur UNE position.
#[derive(Clone)]
struct Analyse {
    /// Évaluation en centipions CÔTÉ BLANCS (mats à ±(32000 − n)).
    eval_blancs: i32,
    /// Meilleur coup en SAN ; vide sur une position terminale.
    meilleur: String,
}

fn position_de_fen(fen: &str) -> Option<Chess> {
    fen.parse::<Fen>()
        .ok()?
        .into_position(CastlingMode::Standard)
        .ok()
}

/// Analyse une position, via le cache (indexé par FEN : c'est lui qui fait
/// qu'une position évaluée comme « après » du pli i ne le sera pas à nouveau
/// comme « avant » du pli i+1).
///
/// Les positions TERMINALES ne coûtent rien au moteur : mat = −32000 pour le
/// trait, pat/nulle = 0, et aucun meilleur coup (Stockfish y répondrait
/// « bestmove (none) », une erreur pour src/uci.rs).
///
/// Moteur mort ou muet : il est relancé et l'analyse retentée (ESSAIS_MOTEUR
/// fois) ; si rien n'y fait, None — l'appelant n'annote pas ce tour-ci et
/// réessaiera, plutôt que d'inscrire une évaluation inventée.
fn analyse(
    moteur: &mut Option<UciEngine>,
    opt: &Opt,
    cache: &mut HashMap<String, Analyse>,
    fen: &str,
) -> Option<Analyse> {
    if let Some(a) = cache.get(fen) {
        return Some(a.clone());
    }
    let Some(pos) = position_de_fen(fen) else {
        // Sans ce message, l'arbitre se figerait à 0 % de CPU sur ce pli sans
        // que rien n'en dise la raison (le pli est ensuite abandonné par
        // `traite_pli`, qui compte les échecs).
        eprintln!("arbitre : FEN illisible, position ignorée : {fen}");
        return None;
    };
    let trait_blanc = pos.turn() == Color::White;

    if pos.legal_moves().is_empty() {
        let cp_trait = if pos.is_check() { -MAT_CP } else { 0 };
        let a = Analyse {
            eval_blancs: vers_blancs(cp_trait, trait_blanc),
            meilleur: String::new(),
        };
        cache.insert(fen.to_string(), a.clone());
        return Some(a);
    }

    for essai in 0..ESSAIS_MOTEUR {
        if arret_demande() {
            return None;
        }
        if moteur.is_none() {
            *moteur = ouvre_moteur(opt);
            if moteur.is_none() {
                dors(1_000);
                continue;
            }
        }
        let m = moteur.as_mut().expect("moteur ouvert juste au-dessus");
        match m.meilleur_coup_et_score_brut_fen(fen, opt.movetime) {
            Ok((coup_uci, Some(score))) => {
                let meilleur = UciMove::from_ascii(coup_uci.as_bytes())
                    .ok()
                    .and_then(|u| u.to_move(&pos).ok())
                    .map(|c| SanPlus::from_move(pos.clone(), &c).to_string())
                    .unwrap_or_else(|| {
                        // Coup imparsable ou illégal : on garde l'UCI brut
                        // plutôt que de paniquer — mais on le DIT. Une chaîne
                        // UCI ne pourra jamais égaler un SAN : le coup ne sera
                        // plus jamais classé « meilleur », et le panneau
                        // afficherait « moteur e2e4 » sans explication.
                        eprintln!(
                            "arbitre : coup moteur « {coup_uci} » inconvertible en SAN sur \
                             {fen} — comparaison au coup joué impossible pour cette position"
                        );
                        coup_uci
                    });
                let a = Analyse {
                    eval_blancs: vers_blancs(cp_du_score(score), trait_blanc),
                    meilleur,
                };
                cache.insert(fen.to_string(), a.clone());
                return Some(a);
            }
            // Coup rendu mais AUCUN score : rien à annoter, on retente.
            Ok((_, None)) => {
                eprintln!("arbitre : aucun score sur {fen} (essai {})", essai + 1);
            }
            Err(e) => {
                eprintln!("arbitre : moteur en échec ({e}) — relance");
                *moteur = None; // Drop → quit/kill, puis réouverture au tour suivant
            }
        }
    }
    None
}

/// Ouvre le moteur d'analyse : PLEINE FORCE (aucun UCI_LimitStrength — c'est
/// l'arbitre, pas un adversaire), bridé en ressources pour ne pas voler la
/// machine au match.
fn ouvre_moteur(opt: &Opt) -> Option<UciEngine> {
    match UciEngine::lance(&opt.engine) {
        Ok(mut m) => {
            if let Err(e) = m.regle_ressources(opt.threads, opt.hash) {
                eprintln!("arbitre : options du moteur refusées ({e})");
                return None;
            }
            if let Err(e) = m.nouvelle_partie() {
                eprintln!("arbitre : ucinewgame en échec ({e})");
                return None;
            }
            Some(m)
        }
        Err(e) => {
            eprintln!("arbitre : lancement du moteur impossible ({e})");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Sorties
// ---------------------------------------------------------------------------

/// Écriture ATOMIQUE (.tmp puis rename, comme src/direct.rs et match.rs) : le
/// lecteur (serve.exe → match.js) ne voit jamais de JSON partiel. Les erreurs
/// d'E/S sont signalées mais jamais fatales.
fn ecrit_atomique(chemin: &str, contenu: &str) {
    let tmp = format!("{chemin}.tmp");
    // Les deux échecs sont signalés : sans cela, le JSON publié resterait figé
    // sur l'état précédent et la page afficherait indéfiniment des annotations
    // périmées sans rien laisser paraître.
    if let Err(e) = std::fs::write(&tmp, contenu) {
        eprintln!("arbitre : écriture de {tmp} impossible ({e}) — publication sautée");
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, chemin) {
        eprintln!("arbitre : publication impossible ({e})");
        let _ = std::fs::remove_file(&tmp); // pas de .tmp orphelin
    }
}

/// Publie l'état d'arbitrage de la partie en cours.
fn publie(opt: &Opt, live: &Live, annotations: &[Annotation]) {
    let plis: Vec<serde_json::Value> = annotations.iter().map(|a| a.json()).collect();
    let v = serde_json::json!({
        "partie": live.partie,
        "champion_blanc": live.champion_blanc,
        "movetime_ms": opt.movetime,
        "plis": plis,
        "resume": resume(annotations),
    });
    ecrit_atomique(&opt.out, &v.to_string());
}

/// Ajoute une ligne au CSV (créé avec son entête au besoin) et VIDE le tampon :
/// un Ctrl-C ou une coupure ne perd jamais une analyse déjà payée.
fn ajoute_csv(chemin: &str, a: &Annotation) {
    let neuf = !std::path::Path::new(chemin).exists();
    let fichier = std::fs::OpenOptions::new().create(true).append(true).open(chemin);
    match fichier {
        Ok(mut f) => {
            let mut ligne = String::new();
            if neuf {
                ligne.push_str(ENTETE_CSV);
                ligne.push('\n');
            }
            ligne.push_str(&a.ligne_csv());
            ligne.push('\n');
            if let Err(e) = f.write_all(ligne.as_bytes()).and_then(|_| f.flush()) {
                eprintln!("arbitre : écriture CSV impossible ({e})");
            }
        }
        Err(e) => eprintln!("arbitre : ouverture de {chemin} impossible ({e})"),
    }
}

// ---------------------------------------------------------------------------
// Cohérence CSV / match diffusé (le verdict lui-même est dans la bibliothèque,
// echec::arbitre::csv_coherent, pour être testé)
// ---------------------------------------------------------------------------

/// Archive le CSV périmé (renommé avec l'horodatage : on n'efface JAMAIS une
/// analyse déjà payée) et repart d'un fichier neuf.
///
/// Si le renommage échoue (fichier ouvert dans un tableur, verrou antivirus),
/// on ABANDONNE : surtout pas de `write` de l'entête à la place, qui
/// tronquerait le fichier et jetterait précisément les analyses que cette
/// fonction promet de garder. L'arbitre s'arrête et le dit ; l'utilisateur
/// range le fichier et relance.
fn archive_csv(chemin: &str) -> Result<(), String> {
    let horodatage = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let archive = format!("{chemin}.{horodatage}.bak");
    match std::fs::rename(chemin, &archive) {
        Ok(()) => {
            println!("arbitre : CSV périmé archivé dans {archive} — reprise à zéro");
            Ok(())
        }
        Err(e) => Err(format!(
            "archivage de {chemin} impossible ({e}) — rien n'a été touché. \
             Le CSV décrit un autre match que celui diffusé : déplacez-le \
             (ou fermez ce qui le tient ouvert), puis relancez l'arbitre."
        )),
    }
}

/// Verrou d'instance, à côté du CSV : deux arbitres sur le même fichier
/// doubleraient le coût moteur (interdit sur cette machine) et fausseraient le
/// résumé. Rend le chemin du verrou pris, à effacer en sortant ; sort du
/// programme si un autre arbitre le tient déjà. Un système qui refuse la
/// création (droits, disque en lecture seule) ne bloque pas l'arbitre : on le
/// signale et on continue sans garde-fou.
fn prend_verrou(csv: &str) -> Option<String> {
    let chemin = format!("{csv}.lock");
    match std::fs::OpenOptions::new().create_new(true).write(true).open(&chemin) {
        Ok(mut f) => {
            let _ = writeln!(f, "{}", std::process::id());
            Some(chemin)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            let pid = std::fs::read_to_string(&chemin).unwrap_or_default();
            eprintln!(
                "arbitre : un autre arbitre travaille déjà sur {csv} (verrou {chemin}, \
                 PID {}) — sortie sans rien toucher.",
                pid.trim()
            );
            eprintln!("arbitre : si plus aucun arbitre ne tourne, effacez {chemin}.");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("arbitre : verrou {chemin} impossible ({e}) — on continue sans.");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Annotation d'un pli
// ---------------------------------------------------------------------------

/// Ce que la boucle traîne d'un tour à l'autre : le coûteux (moteur, cache
/// d'analyses) et la mémoire (annotations, plis en échec). Regroupé pour que le
/// rattrapage de fin de partie travaille sur le MÊME état que la boucle.
struct Etat {
    moteur: Option<UciEngine>,
    cache: HashMap<String, Analyse>,
    /// Tout le CSV, toutes parties confondues ; l'affichage, lui, ne porte que
    /// sur la partie en cours.
    memoire: Vec<Annotation>,
    /// Plis (partie, ply) définitivement abandonnés — voir `ESSAIS_PLI`.
    abandons: HashSet<(u32, u32)>,
    /// Échecs consécutifs par pli, effacés dès qu'il est annoté.
    echecs: HashMap<(u32, u32), u32>,
}

impl Etat {
    /// Annotations d'une partie, dans l'ordre des plis.
    fn annotations(&self, partie: u32) -> Vec<Annotation> {
        let mut v: Vec<Annotation> = self
            .memoire
            .iter()
            .filter(|a| a.partie == partie)
            .cloned()
            .collect();
        v.sort_by_key(|a| a.ply);
        v
    }

    /// Plis abandonnés de cette partie, à sauter comme s'ils étaient annotés.
    fn ignores(&self, partie: u32) -> Vec<u32> {
        self.abandons
            .iter()
            .filter(|(p, _)| *p == partie)
            .map(|(_, pli)| *pli)
            .collect()
    }
}

/// Annote le demi-coup `pli` (1-based) de `live`. None si le moteur n'a rien
/// rendu d'exploitable ou si l'arrêt a été demandé — rien n'est alors inscrit.
fn annote(opt: &Opt, live: &Live, pli: u32, etat: &mut Etat) -> Option<Annotation> {
    let i = pli as usize - 1; // index 0-based du coup dans history_san
    let fen_avant = live.fen.get(i)?.clone();
    let fen_apres = live.fen.get(i + 1)?.clone();
    // Deux analyses SÉQUENTIELLES (un seul moteur, un seul emprunt à la fois) ;
    // la seconde est presque toujours servie par le cache au pli suivant —
    // c'est là qu'est l'économie (voir l'en-tête).
    let avant = analyse(&mut etat.moteur, opt, &mut etat.cache, &fen_avant)?;
    let apres = analyse(&mut etat.moteur, opt, &mut etat.cache, &fen_apres)?;
    let trait_blanc = pli % 2 == 1;
    let joue = live.san.get(i)?.clone();
    let perte = perte_cp(avant.eval_blancs, apres.eval_blancs, trait_blanc);
    let identique = !avant.meilleur.is_empty() && avant.meilleur == joue;
    let pieces = position_de_fen(&fen_avant)
        .map(|p| p.board().occupied().count() as u32)
        .unwrap_or(32);
    Some(Annotation {
        partie: live.partie,
        ply: pli,
        camp: camp(pli, live.champion_blanc).to_string(),
        phase: phase(pli, pieces),
        eval_avant: avant.eval_blancs,
        eval_apres: apres.eval_blancs,
        meilleur: avant.meilleur.clone(),
        joue,
        perte_cp: perte,
        classement: classement(perte, identique),
    })
}

/// Annote un pli et l'inscrit (console, CSV, mémoire). Rend `false` si
/// l'analyse a échoué : l'appelant réessaiera, et au bout de `ESSAIS_PLI`
/// échecs le pli est ABANDONNÉ — journalisé avec sa FEN — pour que la boucle
/// avance au lieu de s'acharner indéfiniment sur la même position.
fn traite_pli(opt: &Opt, live: &Live, pli: u32, etat: &mut Etat) -> bool {
    let cle = (live.partie, pli);
    if let Some(a) = annote(opt, live, pli, etat) {
        println!(
            "p{} pli {:>3} {:<9} {:<10} éval {:>+7.2} · moteur {:<7} joué {:<7} \
             perte {:>4} cp → {}",
            a.partie,
            a.ply,
            a.camp,
            a.phase.nom(),
            a.eval_avant as f64 / 100.0,
            if a.meilleur.is_empty() { "—" } else { a.meilleur.as_str() },
            a.joue,
            a.perte_cp,
            a.classement.nom()
        );
        ajoute_csv(&opt.csv, &a);
        etat.memoire.push(a);
        etat.echecs.remove(&cle);
        return true;
    }
    // Arrêt demandé : ce n'est pas un échec du pli, on s'en va sans le noircir.
    if arret_demande() {
        return false;
    }
    let echecs = {
        let c = etat.echecs.entry(cle).or_insert(0);
        *c += 1;
        *c
    };
    if echecs >= ESSAIS_PLI {
        eprintln!(
            "arbitre : pli {pli} de la partie {} ABANDONNÉ après {echecs} tentatives \
             (FEN {}) — on passe au suivant.",
            live.partie,
            live.fen.get(pli as usize - 1).map_or("?", |s| s.as_str())
        );
        etat.abandons.insert(cle);
    }
    false
}

// ---------------------------------------------------------------------------
// Rattrapage de fin de partie
// ---------------------------------------------------------------------------

/// Historique le plus complet dont on dispose pour la partie qui vient de se
/// clore : l'instantané du direct, ou le PGN écrit par match.rs s'il en dit
/// plus. Le PGN n'est retenu que s'il PROLONGE l'instantané (mêmes coups sur
/// le préfixe commun) : un PGN resté d'un match précédent, avec la même
/// numérotation, ne doit jamais faire annoter des coups qui n'ont pas été joués.
fn source_de_rattrapage(opt: &Opt, precedent: &Live) -> Live {
    if opt.pgn.is_empty() {
        return precedent.clone();
    }
    let chemin = format!("{}/partie_{:03}.pgn", opt.pgn, precedent.partie);
    let Ok(texte) = std::fs::read_to_string(&chemin) else {
        return precedent.clone();
    };
    let san = coups_du_pgn(&texte);
    let fen = fens_de_coups(&san);
    if fen.len() != san.len() + 1 {
        eprintln!(
            "arbitre : {chemin} — seulement {} coup(s) rejoué(s) sur {} ; on s'en tient \
             au direct.",
            fen.len() - 1,
            san.len()
        );
        return precedent.clone();
    }
    let du_pgn = Live {
        partie: precedent.partie,
        champion_blanc: precedent.champion_blanc,
        termine: precedent.termine,
        san,
        fen,
    };
    let prolonge = du_pgn.plis() > precedent.plis()
        && precedent
            .san
            .iter()
            .zip(du_pgn.san.iter())
            .all(|(a, b)| a == b);
    if !prolonge {
        return precedent.clone();
    }
    println!(
        "arbitre : {chemin} complète le direct ({} plis contre {}).",
        du_pgn.plis(),
        precedent.plis()
    );
    du_pgn
}

/// Termine l'annotation de la partie qui vient de se clore, AVANT de suivre la
/// suivante.
///
/// match.rs n'insère aucune pause entre deux parties : la publication qui porte
/// le résultat est suivie de l'écriture du PGN puis, dans la foulée, du premier
/// état de la partie suivante avec un historique VIDE. Sans ce rattrapage, les
/// un ou deux derniers plis de chaque partie — le mat, la gaffe décisive — ne
/// seraient jamais annotés et rien ne pourrait plus les retrouver.
///
/// Rien n'est publié ici : la page suit déjà la partie suivante, et
/// models/match_arbitre.json ne décrit qu'une partie à la fois. Le CSV, lui,
/// reçoit tout — c'est la mémoire durable.
fn rattrape_fin_de_partie(opt: &Opt, precedent: &Live, etat: &mut Etat) {
    let source = source_de_rattrapage(opt, precedent);
    let total = source.plis();
    let mut annonce = false;
    while !arret_demande() {
        let annotations = etat.annotations(source.partie);
        let pli = prochain_pli_hors(&annotations, source.partie, &etat.ignores(source.partie));
        if pli as usize > total {
            break;
        }
        if !annonce {
            println!(
                "arbitre : partie {} close — rattrapage des plis {pli} à {total} avant de \
                 passer à la suivante.",
                source.partie
            );
            annonce = true;
        }
        if traite_pli(opt, &source, pli, etat) {
            dors(opt.pause);
        } else {
            dors(ATTENTE_MS);
        }
    }
    // Le cache de la partie close ne resservira plus : c'est l'appelant qui le
    // vide, une fois le rattrapage fini.
}

// ---------------------------------------------------------------------------
// Boucle principale
// ---------------------------------------------------------------------------

/// Rend le code de sortie du processus (0 = fin normale).
fn boucle(opt: &Opt) -> i32 {
    let mut etat = Etat {
        moteur: None,
        cache: HashMap::new(),
        memoire: std::fs::read_to_string(&opt.csv)
            .map(|c| lit_csv(&c))
            .unwrap_or_default(),
        abandons: HashSet::new(),
        echecs: HashMap::new(),
    };
    if !etat.memoire.is_empty() {
        println!("arbitre : reprise — {} pli(s) déjà annoté(s)", etat.memoire.len());
    }

    // Dernière publication : (partie, nombre de plis publiés) — évite de
    // réécrire le JSON à chaque tour de sondage quand rien n'a changé.
    let mut publie_pour: Option<(u32, usize)> = None;
    let mut partie_courante: u32 = 0;
    // Dernier état vu de la partie en cours : c'est la source de rattrapage
    // quand match.exe passe à la partie suivante.
    let mut dernier_live: Option<Live> = None;

    while !arret_demande() {
        let Some(live) = lit_live(&opt.live) else {
            dors(ATTENTE_MS);
            continue;
        };
        if live.partie != partie_courante {
            if let Some(precedent) = dernier_live.take() {
                rattrape_fin_de_partie(opt, &precedent, &mut etat);
            }
            partie_courante = live.partie;
            // Le cache de positions de la partie close ne resservira jamais
            // (et le moteur repart sur une TT propre).
            etat.cache.clear();
            if let Some(m) = etat.moteur.as_mut() {
                let _ = m.nouvelle_partie();
            }
        }
        dernier_live = Some(live.clone());

        let mut annotations = etat.annotations(live.partie);

        // Le CSV décrit-il bien ce match ? Sinon on l'archive et on repart.
        if !csv_coherent(&annotations, &live.san, live.plis()) {
            if let Err(e) = archive_csv(&opt.csv) {
                eprintln!("arbitre : {e}");
                return 1;
            }
            etat.memoire.clear();
            etat.abandons.clear();
            etat.echecs.clear();
            annotations.clear();
            publie_pour = None;
        }

        let pli = prochain_pli_hors(&annotations, live.partie, &etat.ignores(live.partie));
        if pli as usize <= live.plis() {
            if traite_pli(opt, &live, pli, &mut etat) {
                let annotations = etat.annotations(live.partie);
                publie(opt, &live, &annotations);
                publie_pour = Some((live.partie, annotations.len()));
                dors(opt.pause);
            } else {
                // Moteur indisponible ou pli abandonné : on laisse respirer la
                // machine avant de reprendre (jamais de rafale).
                dors(ATTENTE_MS);
            }
            continue;
        }

        // À jour : on publie si quelque chose a changé, puis on sonde.
        if publie_pour != Some((live.partie, annotations.len())) {
            publie(opt, &live, &annotations);
            publie_pour = Some((live.partie, annotations.len()));
        }
        if live.termine {
            println!("arbitre : match terminé et intégralement annoté — sortie.");
            break;
        }
        dors(ATTENTE_MS);
    }

    if arret_demande() {
        println!("arbitre : arrêt demandé — sortie propre.");
    }
    0
    // Drop de `etat.moteur` : « quit » poli puis kill (src/uci.rs) — pas de zombie.
}

fn main() {
    let opt = parse_args();
    arme_ctrl_c();
    println!(
        "arbitre : surveille {} · moteur {} ({} thread(s), {} Mio, movetime {} ms)",
        opt.live, opt.engine, opt.threads, opt.hash, opt.movetime
    );
    println!("arbitre : sorties {} et {}", opt.out, opt.csv);

    let verrou = prend_verrou(&opt.csv);
    let code = boucle(&opt);
    if let Some(v) = verrou {
        let _ = std::fs::remove_file(&v);
    }
    if code != 0 {
        std::process::exit(code);
    }
}
