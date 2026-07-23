//! Serveur web (plateau + dashboard d'entraînement). À IMPLÉMENTER selon ce contrat.
//!
//! Écoute sur 127.0.0.1:8778 ET [::1]:8778 (deux instances tiny_http, threads
//! partageant le même état Arc — leçon du poker : localhost peut résoudre en IPv6
//! d'abord, ne servir qu'IPv4 coûte 2 s par requête).
//!
//! Fichiers statiques depuis ./web : "/" → index.html, "/training" → training.html,
//! sinon chemin relatif dans web/ (Content-Type selon l'extension : html, js, css).
//!
//! Une seule session de jeu (Mutex), comme le serveur poker.
//!
//! API JSON :
//!  GET  /api/state       → état courant (voir schéma)
//!  POST /api/new-game    {"opponent": "random"|"material"|"t1h"|"t3h"|"t10h"|"t30h"|"t100h"|"latest",
//!                         "color": "white"|"black"|"random"}
//!                        → nouvel état ; si l'humain a les noirs, l'IA joue
//!                          immédiatement son premier coup.
//!  POST /api/move        {"uci": "e2e4"} → joue le coup humain (400 si illégal),
//!                        puis la réponse de l'IA si la partie continue ; renvoie l'état.
//!  GET  /api/checkpoints → {"opponents": [{"id": "random", "label": "Aléatoire", "available": true},
//!                          {"id": "material", ...}, {"id": "t1h", "label": "IA — 1 h d'entraînement",
//!                          "available": <fichier présent>}, ..., {"id": "latest", ...}]}
//!  GET  /api/progress    → {"metrics": {"elapsed_hours": [...], "loss": [...],
//!                          "pct_vs_random": [...], "pct_vs_material": [...],
//!                          "games": [...]}, "state": {"trained_secs": ..,
//!                          "games": .., "positions": .., "cycles": ..}}
//!                          (lecture de models/metrics.csv + models/state.json à chaque appel)
//!
//! Schéma de l'état ("state") :
//!  {"fen": "...", "turn": "white"|"black", "your_color": "white"|"black",
//!   "legal": ["e2e4", ...]  // uniquement si c'est à l'humain de jouer, sinon [],
//!   "last_move": "e7e5"|null, "history_san": ["e4", "e5", ...],
//!   "result": null|"1-0"|"0-1"|"1/2-1/2", "result_reason": null|"mat"|"pat"|"50 coups"|
//!             "3 répétitions"|"matériel insuffisant"|"abandon",
//!   "opponent": "t1h", "thinking_ms": 42}
//!
//! Adversaires : "random" → RandomBot ; "material" → MaterialBot(depth 2) ;
//! paliers et "latest" → Mlp chargé + BotRecherche (négamax alpha-bêta,
//! movetime 150 ms par coup, température 0). Le bot vit DANS la session :
//! sa table de transposition est réutilisée entre les coups d'une même
//! partie, et une nouvelle partie crée un nouveau bot.
//! "latest" → models/chess_best.bin (modèle promu par gating) s'il existe,
//! sinon chess_latest.bin ; rechargé à chaud si le mtime du fichier
//! réellement servi change (comme le serveur poker) — le bot de la session
//! est alors recréé avec le nouveau réseau. Un palier absent → 400 avec
//! message clair. L'IA joue avec une petite graine aléatoire à chaque
//! partie (variété).

use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use rand::Rng;
use shakmaty::fen::Fen;
use shakmaty::san::San;
use shakmaty::uci::UciMove;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{CastlingMode, Chess, Color, EnPassantMode, Move, Position};
use tiny_http::{Header, Method, Request, Response, Server};

use echec::bots::{Bot, BotRecherche, MaterialBot, RandomBot};
use echec::checkpoints::MILESTONES_H;
use echec::nn::Mlp;
use echec::search::Limites;

/// Dossier des modèles (relatif au répertoire de lancement, comme l'entraîneur).
const MODELS_DIR: &str = "models";
/// Dossier des fichiers statiques.
const WEB_DIR: &str = "web";
/// Port d'écoute (IPv4 et IPv6).
const PORT: u16 = 8778;
/// Limites de recherche des adversaires réseau : 150 ms par coup,
/// pas de plafond de nœuds ni de profondeur (le temps arbitre).
const LIMITES_SERVEUR: Limites = Limites {
    max_noeuds: 0,
    max_profondeur: 0,
    movetime_ms: 150,
};

// ---------------------------------------------------------------------------
// Session de jeu
// ---------------------------------------------------------------------------

/// Session unique : la partie en cours entre l'humain et l'IA.
struct Session {
    pos: Chess,
    /// Couleur de l'humain.
    your_color: Color,
    /// Identifiant de l'adversaire ("random", "material", "t1h", ..., "latest").
    opponent: String,
    /// Coups joués, en notation SAN (calculée AVANT de jouer chaque coup).
    history_san: Vec<String>,
    /// Dernier coup joué (UCI), tous camps confondus.
    last_move: Option<String>,
    result: Option<&'static str>,
    result_reason: Option<&'static str>,
    /// Occurrences de chaque hachage zobrist vu dans la partie
    /// (détection de la nulle par 3 répétitions, côté serveur aussi).
    repetitions: HashMap<u64, u32>,
    /// Durée de réflexion du dernier coup de l'IA, en millisecondes.
    thinking_ms: u64,
    /// Graine du prochain coup de l'IA (tirée au hasard à chaque partie,
    /// incrémentée à chaque coup pour la variété).
    graine: u64,
    /// Bot de recherche des adversaires réseau, créé au premier coup de l'IA
    /// et conservé toute la partie (sa table de transposition est réutilisée
    /// entre les coups — c'est une grosse part de sa force). On garde aussi
    /// le réseau qui a servi à le créer : si le rechargement à chaud fournit
    /// un nouveau réseau (mtime changé), on recrée le bot.
    bot: Option<(Arc<Mlp>, BotRecherche)>,
}

/// Hachage zobrist de la position courante (en passant légal uniquement).
fn hachage(pos: &Chess) -> u64 {
    pos.zobrist_hash::<Zobrist64>(EnPassantMode::Legal).0
}

impl Session {
    /// Nouvelle partie depuis la position initiale.
    fn nouvelle(opponent: &str, your_color: Color) -> Session {
        let pos = Chess::default();
        let mut repetitions = HashMap::new();
        repetitions.insert(hachage(&pos), 1u32);
        Session {
            pos,
            your_color,
            opponent: opponent.to_string(),
            history_san: Vec::new(),
            last_move: None,
            result: None,
            result_reason: None,
            repetitions,
            thinking_ms: 0,
            graine: rand::thread_rng().gen(),
            // Créé paresseusement au premier coup de l'IA (nouvelle partie
            // = nouveau bot, donc table de transposition vierge).
            bot: None,
        }
    }

    /// Joue un coup supposé légal : SAN calculé AVANT de jouer, puis mise à
    /// jour de l'historique, du suivi zobrist et de l'issue éventuelle
    /// (mat, pat, matériel insuffisant, 50 coups, 3 répétitions).
    fn jouer(&mut self, m: &Move) {
        let san = San::from_move(&self.pos, m).to_string();
        self.pos.play_unchecked(m);
        self.history_san.push(san);
        self.last_move = Some(m.to_uci(CastlingMode::Standard).to_string());

        let compte = {
            let entree = self.repetitions.entry(hachage(&self.pos)).or_insert(0);
            *entree += 1;
            *entree
        };

        if self.pos.is_checkmate() {
            // Le camp au trait est maté : le camp qui vient de jouer gagne.
            self.result = Some(match self.pos.turn() {
                Color::White => "0-1",
                Color::Black => "1-0",
            });
            self.result_reason = Some("mat");
        } else if self.pos.is_stalemate() {
            self.result = Some("1/2-1/2");
            self.result_reason = Some("pat");
        } else if self.pos.is_insufficient_material() {
            self.result = Some("1/2-1/2");
            self.result_reason = Some("matériel insuffisant");
        } else if self.pos.halfmoves() >= 100 {
            self.result = Some("1/2-1/2");
            self.result_reason = Some("50 coups");
        } else if compte >= 3 {
            self.result = Some("1/2-1/2");
            self.result_reason = Some("3 répétitions");
        }
    }
}

// ---------------------------------------------------------------------------
// Cache des modèles
// ---------------------------------------------------------------------------

/// Cache des réseaux chargés : chemin → (mtime au chargement, réseau partagé).
/// "latest" (et les paliers, par la même mécanique) est rechargé à chaud
/// dès que le mtime du fichier change.
type CacheModeles = Mutex<HashMap<String, (SystemTime, Arc<Mlp>)>>;

/// État global partagé entre les deux listeners (IPv4/IPv6).
struct Etat {
    session: Mutex<Session>,
    cache: CacheModeles,
}

/// Chemin du fichier modèle pour un identifiant d'adversaire réseau,
/// None pour "random"/"material"/inconnu.
///
/// "latest" sert models/chess_best.bin (modèle promu par gating) s'il existe,
/// sinon chess_latest.bin. Réévalué à CHAQUE appel : dès que le gating promeut
/// un premier best, c'est lui qui est servi — et le rechargement à chaud
/// (mtime dans le cache) surveille le fichier réellement servi.
fn chemin_modele(id: &str) -> Option<String> {
    if id == "latest" {
        let best = format!("{MODELS_DIR}/chess_best.bin");
        if Path::new(&best).exists() {
            return Some(best);
        }
        return Some(format!("{MODELS_DIR}/chess_latest.bin"));
    }
    for &h in MILESTONES_H {
        if id == format!("t{}h", h as u64) {
            return Some(format!("{MODELS_DIR}/chess_t{}h.bin", h as u64));
        }
    }
    None
}

/// Charge (ou récupère du cache) le réseau du chemin donné.
/// Recharge si le mtime a changé depuis la mise en cache.
fn charger_modele(cache: &CacheModeles, chemin: &str) -> Result<Arc<Mlp>, String> {
    let mtime = std::fs::metadata(chemin)
        .and_then(|md| md.modified())
        .map_err(|_| format!("palier indisponible : {chemin} absent (pas encore entraîné ?)"))?;
    let mut cache = cache.lock().unwrap();
    if let Some((ancien, net)) = cache.get(chemin) {
        if *ancien == mtime {
            return Ok(net.clone());
        }
    }
    let net = Arc::new(
        Mlp::load(chemin).map_err(|e| format!("échec de chargement de {chemin} : {e}"))?,
    );
    cache.insert(chemin.to_string(), (mtime, net.clone()));
    Ok(net)
}

// ---------------------------------------------------------------------------
// Coup de l'IA
// ---------------------------------------------------------------------------

/// Fait jouer l'adversaire de la session (si la partie continue).
/// Mesure thinking_ms autour du choix du coup.
fn coup_ia(session: &mut Session, cache: &CacheModeles) -> Result<(), String> {
    if session.result.is_some() {
        return Ok(());
    }
    let graine = session.graine;
    session.graine = session.graine.wrapping_add(1);

    // Pour un adversaire réseau : (re)création du bot AVANT de chronométrer —
    // thinking_ms mesure le coup, pas le chargement du fichier modèle.
    if !matches!(session.opponent.as_str(), "random" | "material") {
        let chemin = chemin_modele(&session.opponent)
            .ok_or_else(|| format!("adversaire inconnu : {}", session.opponent))?;
        match charger_modele(cache, &chemin) {
            Ok(net) => {
                // Bot absent (premier coup de la partie) ou réseau rechargé à
                // chaud (nouveau Arc dans le cache) → nouveau bot, TT vierge.
                let recreer = match &session.bot {
                    Some((ancien, _)) => !Arc::ptr_eq(ancien, &net),
                    None => true,
                };
                if recreer {
                    session.bot = Some((
                        net.clone(),
                        // Température 0 : meilleur coup de la recherche (la
                        // force est l'objectif) ; la variété vient de la
                        // graine de la partie.
                        BotRecherche::new(net, graine, LIMITES_SERVEUR, 0.0),
                    ));
                }
            }
            // Rechargement impossible (ex. fichier momentanément illisible
            // pendant son remplacement par l'entraîneur) : si un bot de
            // session existe déjà, on continue avec lui — le coup humain est
            // déjà joué et aucun endpoint ne relance le coup IA, une erreur
            // 400 ici laisserait la partie coincée jusqu'à new-game. Le coup
            // suivant retentera le rechargement. Sans bot existant (premier
            // coup IA de la partie), l'erreur remonte : rien pour jouer.
            Err(e) => {
                if session.bot.is_none() {
                    return Err(e);
                }
            }
        }
    }

    let debut = Instant::now();
    let coup = match session.opponent.as_str() {
        "random" => RandomBot::new(graine).choose(&session.pos),
        "material" => MaterialBot::new(graine, 2).choose(&session.pos),
        _ => {
            // Bot garanti présent : créé ci-dessus, ou conservé du coup
            // précédent si le rechargement à chaud vient d'échouer.
            let (_, bot) = session.bot.as_mut().expect("bot présent à ce stade");
            bot.choose(&session.pos)
        }
    };
    session.thinking_ms = debut.elapsed().as_millis() as u64;

    match coup {
        Some(m) => {
            session.jouer(&m);
            Ok(())
        }
        // Ne devrait pas arriver : la fin de partie est détectée après chaque coup.
        None => Err("l'IA n'a trouvé aucun coup légal".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Sérialisation JSON
// ---------------------------------------------------------------------------

fn couleur_str(c: Color) -> &'static str {
    match c {
        Color::White => "white",
        Color::Black => "black",
    }
}

/// État courant de la session au schéma du contrat.
fn etat_json(session: &Session) -> serde_json::Value {
    let humain_au_trait =
        session.result.is_none() && session.pos.turn() == session.your_color;
    let legal: Vec<String> = if humain_au_trait {
        session
            .pos
            .legal_moves()
            .iter()
            .map(|m| m.to_uci(CastlingMode::Standard).to_string())
            .collect()
    } else {
        Vec::new()
    };
    serde_json::json!({
        "fen": Fen::from_position(session.pos.clone(), EnPassantMode::Legal).to_string(),
        "turn": couleur_str(session.pos.turn()),
        "your_color": couleur_str(session.your_color),
        "legal": legal,
        "last_move": session.last_move,
        "history_san": session.history_san,
        "result": session.result,
        "result_reason": session.result_reason,
        "opponent": session.opponent,
        "thinking_ms": session.thinking_ms,
    })
}

/// Liste des adversaires avec disponibilité (fichier présent pour les réseaux).
fn checkpoints_json() -> serde_json::Value {
    let mut opponents = vec![
        serde_json::json!({"id": "random", "label": "Aléatoire", "available": true}),
        serde_json::json!({"id": "material", "label": "Matériel (profondeur 2)", "available": true}),
    ];
    for &h in MILESTONES_H {
        let h = h as u64;
        let chemin = format!("{MODELS_DIR}/chess_t{h}h.bin");
        opponents.push(serde_json::json!({
            "id": format!("t{h}h"),
            "label": format!("IA — {h} h d'entraînement"),
            "available": Path::new(&chemin).exists(),
        }));
    }
    // "latest" : chess_best.bin (promu par gating) prioritaire, sinon
    // chess_latest.bin — le libellé dit lequel est réellement servi.
    let best_present = Path::new(&format!("{MODELS_DIR}/chess_best.bin")).exists();
    let latest_present = Path::new(&format!("{MODELS_DIR}/chess_latest.bin")).exists();
    opponents.push(serde_json::json!({
        "id": "latest",
        "label": if best_present {
            "IA — dernier modèle (gating)"
        } else {
            "IA — dernier modèle"
        },
        "available": best_present || latest_present,
    }));
    serde_json::json!({"opponents": opponents})
}

/// Courbes d'entraînement : relit models/metrics.csv et models/state.json
/// à chaque appel (fichiers écrits par l'entraîneur, éventuellement en cours).
fn progress_json() -> serde_json::Value {
    let mut elapsed_hours: Vec<f64> = Vec::new();
    let mut loss: Vec<f64> = Vec::new();
    let mut pct_vs_random: Vec<f64> = Vec::new();
    let mut pct_vs_material: Vec<f64> = Vec::new();
    let mut games: Vec<f64> = Vec::new();

    if let Ok(contenu) = std::fs::read_to_string(format!("{MODELS_DIR}/metrics.csv")) {
        // Entête : elapsed_hours,games,positions,loss,pct_vs_random,pct_vs_material
        for ligne in contenu.lines().skip(1) {
            let cols: Vec<&str> = ligne.split(',').collect();
            if cols.len() < 6 {
                continue; // ligne tronquée (écriture en cours) : ignorée
            }
            let vals: Vec<Option<f64>> =
                cols.iter().map(|c| c.trim().parse::<f64>().ok()).collect();
            if let (Some(eh), Some(g), Some(_p), Some(l), Some(pr), Some(pm)) =
                (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])
            {
                elapsed_hours.push(eh);
                games.push(g);
                loss.push(l);
                pct_vs_random.push(pr);
                pct_vs_material.push(pm);
            }
        }
    }

    // Estimations Elo (fichier écrit par l'entraîneur, absent au tout début).
    let mut elo_hours: Vec<f64> = Vec::new();
    let mut elo_vals: Vec<f64> = Vec::new();
    if let Ok(contenu) = std::fs::read_to_string(format!("{MODELS_DIR}/elo.csv")) {
        // Entête : elapsed_hours,elo
        for ligne in contenu.lines().skip(1) {
            let cols: Vec<&str> = ligne.split(',').collect();
            if cols.len() < 2 {
                continue;
            }
            if let (Ok(h), Ok(e)) = (cols[0].trim().parse::<f64>(), cols[1].trim().parse::<f64>()) {
                elo_hours.push(h);
                elo_vals.push(e);
            }
        }
    }

    // Journal de gating (duels candidat vs champion) et événements de régime
    // (fichiers écrits par l'entraîneur v2, absents avant lui).
    let mut gat_hours: Vec<f64> = Vec::new();
    let mut gat_score: Vec<f64> = Vec::new();
    let mut gat_promu: Vec<f64> = Vec::new();
    if let Ok(contenu) = std::fs::read_to_string(format!("{MODELS_DIR}/gating.csv")) {
        // Entête : elapsed_hours,score_pct,promu
        for ligne in contenu.lines().skip(1) {
            let cols: Vec<&str> = ligne.split(',').collect();
            if cols.len() < 3 {
                continue;
            }
            if let (Ok(h), Ok(s), Ok(p)) = (
                cols[0].trim().parse::<f64>(),
                cols[1].trim().parse::<f64>(),
                cols[2].trim().parse::<f64>(),
            ) {
                gat_hours.push(h);
                gat_score.push(s);
                gat_promu.push(p);
            }
        }
    }
    let mut events: Vec<serde_json::Value> = Vec::new();
    if let Ok(contenu) = std::fs::read_to_string(format!("{MODELS_DIR}/events.csv")) {
        // Entête : elapsed_hours,label
        for ligne in contenu.lines().skip(1) {
            let cols: Vec<&str> = ligne.split(',').collect();
            if cols.len() < 2 {
                continue;
            }
            if let Ok(h) = cols[0].trim().parse::<f64>() {
                events.push(serde_json::json!({"h": h, "label": cols[1].trim()}));
            }
        }
    }

    let state_val: serde_json::Value =
        std::fs::read_to_string(format!("{MODELS_DIR}/state.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
    let champ = |nom: &str| -> serde_json::Value {
        state_val
            .get(nom)
            .cloned()
            .unwrap_or(serde_json::json!(0))
    };

    serde_json::json!({
        "metrics": {
            "elapsed_hours": elapsed_hours,
            "loss": loss,
            "pct_vs_random": pct_vs_random,
            "pct_vs_material": pct_vs_material,
            "games": games,
        },
        "elo": {
            "elapsed_hours": elo_hours,
            "elo": elo_vals,
        },
        "gating": {
            "elapsed_hours": gat_hours,
            "score_pct": gat_score,
            "promu": gat_promu,
        },
        "events": events,
        "state": {
            "trained_secs": champ("trained_secs"),
            "games": champ("games"),
            "positions": champ("positions"),
            "cycles": champ("cycles"),
        },
    })
}

// ---------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------

type Rep = Response<Cursor<Vec<u8>>>;

fn entete(nom: &str, valeur: &str) -> Header {
    Header::from_bytes(nom.as_bytes(), valeur.as_bytes()).expect("entête HTTP invalide")
}

fn reponse_json(code: u16, v: &serde_json::Value) -> Rep {
    Response::from_string(v.to_string())
        .with_header(entete("Content-Type", "application/json; charset=utf-8"))
        .with_status_code(code)
}

fn erreur_json(code: u16, message: &str) -> Rep {
    reponse_json(code, &serde_json::json!({"error": message}))
}

/// Fichier statique depuis ./web ("/" → index.html, "/training" → training.html).
fn servir_statique(chemin: &str) -> Rep {
    let relatif = match chemin {
        "/" => "index.html",
        "/training" => "training.html",
        autre => autre.trim_start_matches('/'),
    };
    // Pas de traversée de répertoire.
    if relatif.contains("..") || relatif.contains('\\') {
        return erreur_404();
    }
    let complet = format!("{WEB_DIR}/{relatif}");
    match std::fs::read(&complet) {
        Ok(donnees) => {
            let ext = relatif.rsplit('.').next().unwrap_or("");
            let content_type = match ext {
                "html" => "text/html; charset=utf-8",
                "js" => "application/javascript; charset=utf-8",
                "css" => "text/css; charset=utf-8",
                "json" => "application/json; charset=utf-8",
                "svg" => "image/svg+xml",
                "png" => "image/png",
                "ico" => "image/x-icon",
                _ => "application/octet-stream",
            };
            Response::from_data(donnees).with_header(entete("Content-Type", content_type))
        }
        Err(_) => erreur_404(),
    }
}

fn erreur_404() -> Rep {
    Response::from_string("404 — introuvable")
        .with_header(entete("Content-Type", "text/plain; charset=utf-8"))
        .with_status_code(404)
}

/// POST /api/new-game
fn api_new_game(etat: &Etat, corps: &str) -> Rep {
    let v: serde_json::Value = serde_json::from_str(corps).unwrap_or(serde_json::Value::Null);
    let opponent = match v.get("opponent").and_then(|x| x.as_str()) {
        Some(id) => id.to_string(),
        None => return erreur_json(400, "champ \"opponent\" manquant"),
    };

    // Validation de l'adversaire ; pour un réseau, chargement immédiat
    // (message clair tout de suite si le palier est absent).
    match opponent.as_str() {
        "random" | "material" => {}
        id => match chemin_modele(id) {
            Some(chemin) => {
                if let Err(e) = charger_modele(&etat.cache, &chemin) {
                    return erreur_json(400, &e);
                }
            }
            None => return erreur_json(400, &format!("adversaire inconnu : {id}")),
        },
    }

    let your_color = match v.get("color").and_then(|x| x.as_str()).unwrap_or("random") {
        "white" => Color::White,
        "black" => Color::Black,
        _ => {
            if rand::thread_rng().gen::<bool>() {
                Color::White
            } else {
                Color::Black
            }
        }
    };

    let mut session = Session::nouvelle(&opponent, your_color);
    // Si l'humain a les noirs, l'IA (blancs) joue immédiatement.
    if your_color == Color::Black {
        if let Err(e) = coup_ia(&mut session, &etat.cache) {
            return erreur_json(400, &e);
        }
    }
    let json = etat_json(&session);
    *etat.session.lock().unwrap() = session;
    reponse_json(200, &json)
}

/// POST /api/move
fn api_move(etat: &Etat, corps: &str) -> Rep {
    let v: serde_json::Value = serde_json::from_str(corps).unwrap_or(serde_json::Value::Null);
    let uci_txt = match v.get("uci").and_then(|x| x.as_str()) {
        Some(u) => u,
        None => return erreur_json(400, "champ \"uci\" manquant"),
    };

    let mut session = etat.session.lock().unwrap();
    if session.result.is_some() {
        return erreur_json(400, "la partie est terminée : lancez une nouvelle partie");
    }
    if session.pos.turn() != session.your_color {
        return erreur_json(400, "ce n'est pas à vous de jouer");
    }

    let uci = match UciMove::from_ascii(uci_txt.as_bytes()) {
        Ok(u) => u,
        Err(_) => return erreur_json(400, &format!("coup UCI illisible : {uci_txt}")),
    };
    let m = match uci.to_move(&session.pos) {
        Ok(m) => m,
        Err(_) => return erreur_json(400, &format!("coup illégal : {uci_txt}")),
    };

    session.jouer(&m);
    if session.result.is_none() {
        if let Err(e) = coup_ia(&mut session, &etat.cache) {
            return erreur_json(400, &e);
        }
    }
    reponse_json(200, &etat_json(&session))
}

/// Route une requête et lui répond.
fn traiter(mut req: Request, etat: &Etat) {
    let chemin = req
        .url()
        .split('?')
        .next()
        .unwrap_or("/")
        .to_string();
    let est_get = *req.method() == Method::Get;
    let est_post = *req.method() == Method::Post;

    // Corps lu avant le routage (req est empruntée mutablement ici).
    let corps = if est_post {
        let mut s = String::new();
        let _ = req.as_reader().read_to_string(&mut s);
        s
    } else {
        String::new()
    };

    let reponse: Rep = match chemin.as_str() {
        "/api/state" if est_get => reponse_json(200, &etat_json(&etat.session.lock().unwrap())),
        "/api/checkpoints" if est_get => reponse_json(200, &checkpoints_json()),
        "/api/progress" if est_get => reponse_json(200, &progress_json()),
        "/api/new-game" if est_post => api_new_game(etat, &corps),
        "/api/move" if est_post => api_move(etat, &corps),
        c if c.starts_with("/api/") => erreur_json(404, "route API inconnue"),
        c if est_get => servir_statique(c),
        _ => erreur_404(),
    };
    let _ = req.respond(reponse);
}

/// Boucle de service d'un listener.
fn boucle(serveur: Server, etat: Arc<Etat>) {
    for req in serveur.incoming_requests() {
        traiter(req, &etat);
    }
}

fn main() {
    let etat = Arc::new(Etat {
        // Session par défaut : humain (blancs) contre le bot aléatoire.
        session: Mutex::new(Session::nouvelle("random", Color::White)),
        cache: Mutex::new(HashMap::new()),
    });

    // Listener IPv6 optionnel : échec silencieux si IPv6 indisponible.
    if let Ok(srv6) = Server::http(format!("[::1]:{PORT}")) {
        let etat6 = etat.clone();
        std::thread::spawn(move || boucle(srv6, etat6));
    }

    let srv4 = Server::http(format!("127.0.0.1:{PORT}"))
        .unwrap_or_else(|e| panic!("impossible d'écouter sur 127.0.0.1:{PORT} : {e}"));
    println!("Serveur échecs : http://127.0.0.1:{PORT} (plateau) — /training (courbes)");
    boucle(srv4, etat);
}
