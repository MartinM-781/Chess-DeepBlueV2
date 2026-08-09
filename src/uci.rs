//! Client UCI minimal pour piloter un moteur externe (Stockfish) :
//! - à force limitée (UCI_LimitStrength + UCI_Elo), afin de recaler l'échelle
//!   Elo maison sur une référence réelle (voir src/bin/calibrate.rs) ;
//! - PLEINE FORCE comme étiqueteur (« oracle ») du self-play d'entraînement :
//!   `lance_pleine_force` + `evalue_fen` (voir selfplay.rs et bin/train.rs).
//!
//! Points de vigilance couverts ici :
//! - lecture ligne à ligne UNIQUEMENT quand une réponse est attendue
//!   (uciok, readyok, bestmove) : les `info ...` émis pendant `go` sont drainés
//!   dans la même boucle, donc pas de deadlock tuyau plein ;
//! - ÉCHÉANCE sur chaque lecture (thread lecteur + canal, `recv_timeout`) :
//!   un moteur vivant mais FIGÉ (blocage interne, processus suspendu) est tué
//!   et signalé en erreur au lieu de bloquer un ouvrier de self-play pour la
//!   nuit — l'appelant se replie (repli élève, voir selfplay.rs) et le pool
//!   de train.rs remplace le moteur au prochain emprunt ;
//! - stderr redirigé vers null (un moteur bavard ne peut pas se bloquer dessus) ;
//! - EOF détecté (moteur mort → erreur propre, jamais de boucle infinie) ;
//! - bornes UCI_Elo lues dans la sortie de `uci` et clamp systématique ;
//! - coups de promotion à 5 caractères acceptés (parse via shakmaty::uci) ;
//! - FEN générée avec EnPassantMode::Legal (case e.p. présente seulement si
//!   une prise en passant est réellement légale) ;
//! - Drop : `quit` poli, attente courte, puis kill — pas de processus zombie.

use std::io::{BufRead, BufReader, Error, ErrorKind, Result, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{Chess, EnPassantMode, Move};

use crate::bots::Bot;

/// Bornes UCI_Elo par défaut si le moteur ne les annonce pas (valeurs de
/// Stockfish 18, vérifiées) ; écrasées par le parse de la sortie de `uci`.
const ELO_MIN_DEFAUT: u32 = 1320;
const ELO_MAX_DEFAUT: u32 = 3190;

/// Attente maximale de la fin du processus après `quit` avant kill.
const DELAI_QUIT: Duration = Duration::from_millis(500);

/// Échéance de lecture par LIGNE attendue du moteur (voir `ligne_avant`) —
/// plancher très généreux : même machine saturée (pool de moteurs + rayon au
/// complet), un moteur sain répond en millisecondes ; seul un moteur FIGÉ la
/// dépasse. Un faux positif coûterait un moteur (relancé) ; un vrai blocage
/// non détecté coûterait la nuit d'entraînement.
const DELAI_REPONSE: Duration = Duration::from_secs(10);

/// Échéance adaptée à une réponse de `go movetime N` : plancher DELAI_REPONSE,
/// étiré à 10×N quand le movetime demandé est long (calibration).
fn delai_go(movetime_ms: u64) -> Duration {
    DELAI_REPONSE.max(Duration::from_millis(movetime_ms.saturating_mul(10)))
}

/// Moteur UCI externe (processus enfant + tuyaux).
pub struct UciEngine {
    enfant: Child,
    entree: ChildStdin,
    /// Lignes de stdout, lues par le thread `lecteur` et reçues ici : le canal
    /// permet une échéance (`recv_timeout`) là où un `read_line` direct
    /// bloquerait pour toujours face à un moteur vivant mais figé.
    lignes: mpsc::Receiver<String>,
    /// Thread lecteur de stdout — rejoint au Drop, après la mort du processus
    /// (l'EOF termine sa boucle : jointure bornée, aucun thread fuité).
    lecteur: Option<JoinHandle<()>>,
    /// Moteur condamné (échéance dépassée → kill, ou EOF) : toute lecture
    /// ultérieure échoue immédiatement, sans consommer d'éventuels restes du
    /// canal (une vieille ligne ne doit jamais acquitter une commande neuve).
    mort: bool,
    /// Bornes du spin UCI_Elo annoncées par le moteur.
    pub elo_min: u32,
    pub elo_max: u32,
    /// Budget de réflexion (ms) de chaque `evalue_fen` — fixé par
    /// `lance_pleine_force` (défaut prudent pour un moteur venu de `lance`).
    movetime_ms: u32,
}

/// Erreur d'E/S étiquetée (contexte du protocole UCI).
fn erreur(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Lit stdout du moteur ligne à ligne dans un thread dédié et pousse chaque
/// ligne dans un canal : le fil appelant peut alors imposer une échéance
/// (`recv_timeout`, voir `ligne_avant`) — impossible avec un `read_line`
/// direct, qui bloquerait pour toujours face à un moteur vivant mais figé.
/// EOF ou erreur de lecture → fin du thread → canal déconnecté, détecté par
/// l'appelant comme « moteur mort ». `send` sur un canal non borné ne bloque
/// jamais : le thread se termine toujours une fois le processus mort.
fn demarre_lecteur(sortie: ChildStdout) -> (mpsc::Receiver<String>, JoinHandle<()>) {
    let (emetteur, recepteur) = mpsc::channel();
    let lecteur = std::thread::spawn(move || {
        let mut sortie = BufReader::new(sortie);
        loop {
            let mut ligne = String::new();
            match sortie.read_line(&mut ligne) {
                Ok(0) | Err(_) => return, // EOF ou tuyau cassé : moteur mort
                Ok(_) => {
                    if emetteur.send(ligne).is_err() {
                        return; // récepteur disparu (UciEngine droppé)
                    }
                }
            }
        }
    });
    (recepteur, lecteur)
}

impl UciEngine {
    /// Lance le moteur et fait le handshake `uci` → `uciok`, en relevant les
    /// bornes du spin UCI_Elo au passage.
    pub fn lance(chemin: &str) -> Result<UciEngine> {
        UciEngine::lance_avec_delai(chemin, DELAI_REPONSE)
    }

    /// Cœur de `lance`, échéance de handshake paramétrable — les tests d'un
    /// « moteur » figé ou mort n'attendent pas DELAI_REPONSE.
    fn lance_avec_delai(chemin: &str, delai: Duration) -> Result<UciEngine> {
        let mut enfant = Command::new(chemin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            // stderr → null : le moteur ne peut pas se bloquer sur un tuyau
            // que personne ne lit.
            .stderr(Stdio::null())
            .spawn()?;
        let entree = enfant
            .stdin
            .take()
            .ok_or_else(|| erreur("stdin du moteur indisponible".into()))?;
        let sortie = enfant
            .stdout
            .take()
            .ok_or_else(|| erreur("stdout du moteur indisponible".into()))?;
        let (lignes, lecteur) = demarre_lecteur(sortie);
        let mut moteur = UciEngine {
            enfant,
            entree,
            lignes,
            lecteur: Some(lecteur),
            mort: false,
            elo_min: ELO_MIN_DEFAUT,
            elo_max: ELO_MAX_DEFAUT,
            movetime_ms: 15,
        };
        moteur.envoie("uci")?;
        // Draine l'en-tête (id, option ...) jusqu'à uciok, en relevant les
        // bornes de « option name UCI_Elo type spin default X min Y max Z ».
        loop {
            let ligne = moteur.ligne_avant(delai)?;
            if ligne.trim() == "uciok" {
                break;
            }
            if ligne.starts_with("option name UCI_Elo ") {
                let mots: Vec<&str> = ligne.split_whitespace().collect();
                for f in mots.windows(2) {
                    match f[0] {
                        "min" => moteur.elo_min = f[1].parse().unwrap_or(ELO_MIN_DEFAUT),
                        "max" => moteur.elo_max = f[1].parse().unwrap_or(ELO_MAX_DEFAUT),
                        _ => {}
                    }
                }
            }
        }
        moteur.pret()?;
        Ok(moteur)
    }

    /// Envoie une commande (une ligne) et vide le tampon immédiatement :
    /// sans flush, le moteur ne verrait jamais la commande (deadlock classique).
    fn envoie(&mut self, commande: &str) -> Result<()> {
        writeln!(self.entree, "{commande}")?;
        self.entree.flush()
    }

    /// Attend la prochaine ligne du moteur, au plus `delai` :
    /// - canal déconnecté (EOF : moteur mort) → erreur propre, comme avant ;
    /// - échéance dépassée (moteur VIVANT mais figé — blocage interne,
    ///   processus suspendu) → kill immédiat + erreur : mieux vaut perdre un
    ///   moteur (le pool de train.rs le remplace au prochain emprunt) qu'un
    ///   ouvrier rayon bloqué qui calerait la nuit d'entraînement en silence.
    /// Un moteur condamné échoue immédiatement, SANS consommer les restes du
    /// canal (une vieille ligne « readyok » ne doit jamais acquitter une
    /// commande postérieure à la condamnation).
    fn ligne_avant(&mut self, delai: Duration) -> Result<String> {
        if self.mort {
            return Err(Error::new(
                ErrorKind::BrokenPipe,
                "moteur UCI condamné (mort ou figé)",
            ));
        }
        match self.lignes.recv_timeout(delai) {
            Ok(ligne) => Ok(ligne),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                self.mort = true;
                Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    "le moteur UCI a fermé sa sortie (processus mort ?)",
                ))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                self.mort = true;
                let _ = self.enfant.kill();
                Err(Error::new(
                    ErrorKind::TimedOut,
                    format!("moteur UCI muet au-delà de {delai:?} : figé, tué"),
                ))
            }
        }
    }

    /// `isready` → attend `readyok` (draine tout le reste, y compris d'éventuels
    /// `info` tardifs).
    pub fn pret(&mut self) -> Result<()> {
        self.envoie("isready")?;
        loop {
            if self.ligne_avant(DELAI_REPONSE)?.trim() == "readyok" {
                return Ok(());
            }
        }
    }

    /// Active la force limitée et fixe UCI_Elo, CLAMPÉ aux bornes annoncées
    /// par le moteur. Renvoie l'Elo effectivement appliqué.
    pub fn limite_force(&mut self, elo: u32) -> Result<u32> {
        let borne = elo.clamp(self.elo_min, self.elo_max);
        self.envoie("setoption name UCI_LimitStrength value true")?;
        self.envoie(&format!("setoption name UCI_Elo value {borne}"))?;
        self.pret()?;
        Ok(borne)
    }

    /// `ucinewgame` + synchronisation.
    pub fn nouvelle_partie(&mut self) -> Result<()> {
        self.envoie("ucinewgame")?;
        self.pret()
    }

    /// Pose les deux options de RESSOURCES standard de tout moteur UCI :
    /// `Threads` (fils de recherche) et `Hash` (Mio de table de transposition).
    ///
    /// Sciemment ABSENT du chemin historique : ni la calibration ni le match ne
    /// l'appellent, donc le Fantôme continue de tourner sur les défauts du
    /// moteur (1 thread, 16 Mio) — changer cela changerait sa FORCE, et donc
    /// la validité des matchs déjà joués. C'est l'arbitre (src/bin/arbitre.rs)
    /// qui s'en sert, dans l'autre sens : se BRIDER à 1 thread pendant que le
    /// match consomme la machine.
    pub fn regle_ressources(&mut self, threads: u32, hash_mio: u32) -> Result<()> {
        self.envoie(&format!("setoption name Threads value {}", threads.max(1)))?;
        self.envoie(&format!("setoption name Hash value {}", hash_mio.max(1)))?;
        self.pret()
    }

    /// Lance le moteur PLEINE FORCE — aucun UCI_LimitStrength : l'étiqueteur
    /// (oracle) du self-play doit évaluer au niveau maximal du moteur, pas au
    /// niveau bridé de la calibration. `movetime_ms` est mémorisé et sert de
    /// budget à chaque `evalue_fen`. Mêmes garanties que `lance` (handshake
    /// uciok + readyok), plus un `ucinewgame` initial.
    pub fn lance_pleine_force(chemin: &str, movetime_ms: u32) -> Result<UciEngine> {
        let mut moteur = UciEngine::lance(chemin)?;
        moteur.movetime_ms = movetime_ms;
        moteur.nouvelle_partie()?;
        Ok(moteur)
    }

    /// Évalue une position : `position fen ...` + `go movetime ...`, draine
    /// les lignes `info` en mémorisant le score de CHACUNE, et renvoie celui
    /// de la DERNIÈRE reçue avant `bestmove` (la plus profonde), converti dans
    /// [-1, 1] par `score_de_ligne_info`.
    ///
    /// CONVENTION UCI CRUCIALE : « score cp X » / « score mate N » sont du
    /// point de vue du CAMP AU TRAIT — exactement notre convention v_racine
    /// interne. AUCUN renversement de signe, ni ici ni chez l'appelant.
    ///
    /// Toute erreur d'E/S (moteur mort, EOF, tuyau cassé, moteur FIGÉ au-delà
    /// de l'échéance de lecture — alors tué) ou absence de score avant
    /// `bestmove` → None, JAMAIS de panique : l'appelant se replie sur sa
    /// propre évaluation et la partie continue.
    pub fn evalue_fen(&mut self, fen: &str) -> Option<f32> {
        self.envoie(&format!("position fen {fen}")).ok()?;
        // max(1) : « go movetime 0 » est indéfini chez certains moteurs.
        self.envoie(&format!("go movetime {}", self.movetime_ms.max(1)))
            .ok()?;
        let mut dernier: Option<f32> = None;
        let delai = delai_go(self.movetime_ms as u64);
        loop {
            let ligne = self.ligne_avant(delai).ok()?;
            if ligne.split_whitespace().next() == Some("bestmove") {
                return dernier;
            }
            if let Some(v) = score_de_ligne_info(&ligne) {
                dernier = Some(v);
            }
        }
    }

    /// Demande le meilleur coup sur une FEN donnée en `movetime_ms` ms.
    /// Renvoie le coup UCI brut (« e2e4 », promotion « e7e8q » à 5 caractères).
    /// Les lignes `info ...` émises pendant la recherche sont consommées sur ce
    /// chemin — c'est ce qui évite le deadlock « le moteur écrit, personne ne
    /// lit » (boucle commune : `meilleur_coup_et_score_brut_fen`).
    pub fn meilleur_coup_fen(&mut self, fen: &str, movetime_ms: u64) -> Result<String> {
        self.meilleur_coup_et_score_fen(fen, movetime_ms)
            .map(|(coup, _)| coup)
    }

    /// Comme `meilleur_coup_fen`, en relevant AUSSI le score de la dernière
    /// ligne « info ... score ... » NON BORNÉE reçue avant `bestmove`,
    /// converti dans [-1, 1] par `valeur_de_score` — convention UCI :
    /// point de vue du CAMP AU TRAIT, aucun renversement de signe. None si le
    /// moteur n'a émis aucun score exploitable. Sert au harnais de match
    /// (src/bin/match.rs) pour afficher l'évaluation de l'adversaire sans
    /// requête supplémentaire.
    pub fn meilleur_coup_et_score_fen(
        &mut self,
        fen: &str,
        movetime_ms: u64,
    ) -> Result<(String, Option<f32>)> {
        self.meilleur_coup_et_score_brut_fen(fen, movetime_ms)
            .map(|(coup, score)| (coup, score.map(valeur_de_score)))
    }

    /// Cœur commun de `meilleur_coup_fen` / `meilleur_coup_et_score_fen`, en
    /// rendant le score BRUT (`ScoreUci`) au lieu de l'écrasement dans [-1, 1] :
    /// l'arbitre (src/bin/arbitre.rs) annote des pertes en CENTIPIONS, or
    /// tanh(cp/300) est irréversible et sature : 400 cp donne 0,87 et 900 cp
    /// 0,995 — deux positions très différentes se retrouvent à un cheveu l'une
    /// de l'autre —, et « mat en 7 » ne se distingue plus de « mat en 1 ».
    /// Convention UCI inchangée : point de vue du CAMP AU TRAIT, aucun
    /// renversement de signe ici.
    pub fn meilleur_coup_et_score_brut_fen(
        &mut self,
        fen: &str,
        movetime_ms: u64,
    ) -> Result<(String, Option<ScoreUci>)> {
        self.meilleur_coup_et_score_brut_historique(fen, &[], movetime_ms)
    }

    /// Comme ci-dessus, mais en DÉCLARANT l'historique de la partie : les coups
    /// UCI joués depuis la position `fen_initiale`. Sans lui (`position fen`
    /// nu), le moteur ne voit qu'une position isolée et ne peut PAS détecter
    /// les répétitions de la partie : un camp gagnant y concède la nulle sans
    /// s'en apercevoir — trois de nos quatre premières parties de match se sont
    /// terminées ainsi. `coups` vide → exactement l'ancien comportement.
    /// Version « valeur dans [-1, 1] » de `..._brut_historique` (même
    /// conversion que `meilleur_coup_et_score_fen`).
    pub fn meilleur_coup_et_score_historique(
        &mut self,
        fen_initiale: &str,
        coups: &[String],
        movetime_ms: u64,
    ) -> Result<(String, Option<f32>)> {
        self.meilleur_coup_et_score_brut_historique(fen_initiale, coups, movetime_ms)
            .map(|(coup, score)| (coup, score.map(valeur_de_score)))
    }

    pub fn meilleur_coup_et_score_brut_historique(
        &mut self,
        fen_initiale: &str,
        coups: &[String],
        movetime_ms: u64,
    ) -> Result<(String, Option<ScoreUci>)> {
        let fen = if coups.is_empty() {
            format!("position fen {fen_initiale}")
        } else {
            format!("position fen {fen_initiale} moves {}", coups.join(" "))
        };
        self.envoie(&fen)?;
        self.envoie(&format!("go movetime {movetime_ms}"))?;
        let delai = delai_go(movetime_ms);
        // Deux mémoires : le dernier score EXACT et le dernier score BORNÉ
        // (lowerbound/upperbound). On rend l'exact dès qu'il en existe un — une
        // borne n'est pas une évaluation (voir `score_brut_de_ligne_info`) —,
        // et la borne seulement s'il n'y a jamais eu mieux, plutôt que rien.
        let mut exact: Option<ScoreUci> = None;
        let mut borne: Option<ScoreUci> = None;
        loop {
            let ligne = self.ligne_avant(delai)?;
            let mut mots = ligne.split_whitespace();
            if mots.next() == Some("bestmove") {
                let coup = mots
                    .next()
                    .ok_or_else(|| erreur(format!("bestmove sans coup : {ligne:?}")))?;
                if coup == "(none)" {
                    return Err(erreur(format!("bestmove (none) sur {fen}")));
                }
                if !(4..=5).contains(&coup.len()) {
                    return Err(erreur(format!("coup UCI mal formé : {coup:?}")));
                }
                return Ok((coup.to_string(), exact.or(borne)));
            }
            // Ligne info/string : on mémorise l'éventuel score et on draine.
            match score_brut_de_ligne_info(&ligne) {
                Some((v, false)) => exact = Some(v),
                Some((v, true)) => borne = Some(v),
                None => {}
            }
        }
    }
}

/// Score UCI BRUT, du point de vue du CAMP AU TRAIT :
/// - `Cp(x)` : centipions ;
/// - `Mat(n)` : mat en `n` coups — `n > 0` le trait mate, `n <= 0` le trait se
///   fait mater (`Mat(0)` = position déjà matée).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreUci {
    Cp(i32),
    Mat(i32),
}

/// Écrasement historique d'un score brut dans [-1, 1] (échelle des cibles
/// d'entraînement) : tanh(cp/300), mat → ±1. Extrait tel quel de l'ancien
/// `score_de_ligne_info` — aucun changement de comportement pour ses appelants
/// (selfplay, calibration, match).
fn valeur_de_score(s: ScoreUci) -> f32 {
    match s {
        ScoreUci::Cp(cp) => (cp as f32 / 300.0).tanh(),
        ScoreUci::Mat(n) => {
            if n > 0 {
                1.0
            } else {
                -1.0
            }
        }
    }
}

/// Extrait le score BRUT d'une ligne UCI « ... score cp X | score mate N ... »
/// et dit s'il est BORNÉ (« lowerbound »/« upperbound », émis après le nombre
/// quand la fenêtre d'aspiration a échoué). Ligne sans score exploitable → None.
///
/// Un score borné n'est PAS une évaluation : « upperbound X » dit seulement
/// « au plus X », la vraie valeur peut être très en dessous. Tant que le score
/// servait à étiqueter dans [-1, 1], l'approximation était sans conséquence ;
/// depuis que l'arbitre en tire des CENTIPIONS et des pertes, une dernière
/// ligne bornée avant `bestmove` fabriquerait un écart artificiel — donc une
/// « erreur » ou une « gaffe » imaginaire au tableau de bord. Le drapeau permet
/// aux appelants de préférer le dernier score EXACT (voir
/// `meilleur_coup_et_score_brut_fen`).
fn score_brut_de_ligne_info(ligne: &str) -> Option<(ScoreUci, bool)> {
    let mots: Vec<&str> = ligne.split_whitespace().collect();
    let i = mots.iter().position(|m| *m == "score")?;
    let borne = mots[i..]
        .iter()
        .any(|m| *m == "lowerbound" || *m == "upperbound");
    let score = match (mots.get(i + 1)?, mots.get(i + 2)?) {
        (&"cp", x) => x.parse::<i32>().ok().map(ScoreUci::Cp),
        (&"mate", n) => n.parse::<i32>().ok().map(ScoreUci::Mat),
        _ => None,
    }?;
    Some((score, borne))
}

/// Extrait la valeur d'une ligne UCI contenant « score ... », convertie dans
/// [-1, 1] pour coller à l'échelle des cibles d'entraînement :
/// - « score cp X »   → tanh(X/300) — écrasement doux : ±300 cp ≈ ±0.76 ;
/// - « score mate N » → +1.0 si N > 0 (le trait mate), sinon -1.0 (N <= 0 :
///   le trait se fait mater ; « mate 0 » = déjà maté).
///
/// CONVENTION UCI : le score est du point de vue du CAMP AU TRAIT — identique
/// à notre v_racine interne, donc AUCUN renversement de signe.
/// « lowerbound »/« upperbound » sont acceptés tels quels ICI : après
/// tanh(cp/300), une borne d'aspiration ne déplace l'étiquette que de quelques
/// millièmes, et ce chemin sert à ÉTIQUETTER (self-play, calibration), pas à
/// mesurer des centipions. Ligne sans score exploitable → None.
fn score_de_ligne_info(ligne: &str) -> Option<f32> {
    score_brut_de_ligne_info(ligne).map(|(s, _borne)| valeur_de_score(s))
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        // Sortie polie ; si le moteur traîne (ou si l'écriture échoue), kill.
        let _ = self.envoie("quit");
        let debut = Instant::now();
        let mut termine = false;
        loop {
            match self.enfant.try_wait() {
                Ok(Some(_)) => {
                    termine = true; // terminé proprement, pas de zombie
                    break;
                }
                Ok(None) if debut.elapsed() < DELAI_QUIT => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                _ => break,
            }
        }
        if !termine {
            let _ = self.enfant.kill();
            let _ = self.enfant.wait();
        }
        // Le processus est mort : le thread lecteur rencontre l'EOF et se
        // termine de lui-même — jointure bornée, aucun thread fuité.
        if let Some(lecteur) = self.lecteur.take() {
            let _ = lecteur.join();
        }
    }
}

/// Adversaire `Bot` piloté par un moteur UCI à force limitée. Chaque instance
/// possède SON processus moteur (nécessaire pour les duels parallélisés
/// d'arena::score : aucun état partagé entre parties).
pub struct StockfishBot {
    moteur: UciEngine,
    movetime_ms: u64,
}

impl StockfishBot {
    /// Lance un moteur, limite sa force à `elo` (clampé) et ouvre une partie.
    pub fn new(chemin: &str, elo: u32, movetime_ms: u64) -> Result<StockfishBot> {
        let mut moteur = UciEngine::lance(chemin)?;
        moteur.limite_force(elo)?;
        moteur.nouvelle_partie()?;
        Ok(StockfishBot { moteur, movetime_ms })
    }
}

impl Bot for StockfishBot {
    fn choose(&mut self, pos: &Chess) -> Option<Move> {
        // FEN avec case en passant LÉGALE uniquement (mode Legal) : c'est la
        // convention attendue par les moteurs, et elle évite d'annoncer une
        // case e.p. fantôme qui fausserait les tables de hachage du moteur.
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        let texte = self
            .moteur
            .meilleur_coup_fen(&fen, self.movetime_ms)
            .unwrap_or_else(|e| panic!("échec UCI sur {fen} : {e}"));
        // Parse UCI (gère les 5 caractères de promotion), puis validation de
        // légalité contre la position — un coup illégal du moteur fait paniquer
        // la calibration plutôt que de corrompre silencieusement les scores.
        let uci = UciMove::from_ascii(texte.as_bytes())
            .unwrap_or_else(|e| panic!("coup UCI imparsable {texte:?} : {e}"));
        Some(
            uci.to_move(pos)
                .unwrap_or_else(|e| panic!("coup illégal du moteur {texte:?} sur {fen} : {e}")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shakmaty::uci::UciMove;
    use shakmaty::Position;

    /// Chemin du moteur local (les tests marqués #[ignore] le supposent présent ;
    /// cwd des tests = racine du crate).
    const CHEMIN: &str = "engines/stockfish/stockfish-windows-x86-64-avx2.exe";

    /// Handshake réel : uciok, bornes UCI_Elo, clamp, bestmove, promotion,
    /// FEN en passant. Ignoré par défaut (dépend du binaire local) :
    /// `cargo test --lib -- --ignored uci`.
    #[test]
    #[ignore = "nécessite engines/stockfish en local"]
    fn handshake_uci_reel() {
        let mut moteur = UciEngine::lance(CHEMIN).expect("lancement du moteur");
        // Bornes annoncées par Stockfish 18 (vérifiées à la main).
        assert_eq!(moteur.elo_min, 1320, "min UCI_Elo");
        assert_eq!(moteur.elo_max, 3190, "max UCI_Elo");
        // Clamp aux deux bornes.
        assert_eq!(moteur.limite_force(1000).expect("limite basse"), 1320);
        assert_eq!(moteur.limite_force(9999).expect("limite haute"), 3190);
        assert_eq!(moteur.limite_force(1320).expect("limite nominale"), 1320);
        moteur.nouvelle_partie().expect("ucinewgame");

        // Meilleur coup depuis la position initiale : 4 ou 5 caractères et légal.
        let pos = Chess::default();
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        let coup = moteur.meilleur_coup_fen(&fen, 30).expect("bestmove initial");
        let m = UciMove::from_ascii(coup.as_bytes())
            .expect("parse UCI")
            .to_move(&pos)
            .expect("coup légal");
        assert!(pos.legal_moves().contains(&m), "coup hors liste légale : {coup}");

        // Promotion FORCÉE : Rh1 blanc muré (Rh3 noir + pion h2 noir qui
        // couvre g1), seuls coups légaux = promotions b7b8 — le bestmove est
        // à 5 caractères quel que soit le niveau ou la profondeur. (On ne
        // teste PAS le « choix » de promouvoir : sur un K+P libre, un moteur
        // profond peut légitimement préférer un coup de roi, vérifié a1b2.)
        let coup = moteur
            .meilleur_coup_fen("8/1P6/8/8/8/7k/7p/7K w - - 0 1", 30)
            .expect("bestmove promotion");
        assert_eq!(coup.len(), 5, "promotion attendue à 5 caractères : {coup}");
        assert!(coup.starts_with("b7b8"), "promotion b7b8 attendue : {coup}");

        // En passant : après 1.e4 h6 2.e5 d5, la FEN Legal doit annoncer d6
        // et le moteur doit accepter la position (exd6 e.p. est légal).
        let mut pos = Chess::default();
        for uci in ["e2e4", "h7h6", "e4e5", "d7d5"] {
            let m = UciMove::from_ascii(uci.as_bytes())
                .unwrap()
                .to_move(&pos)
                .unwrap();
            pos = pos.play(&m).unwrap();
        }
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        assert!(fen.contains(" d6 "), "case e.p. d6 absente de la FEN : {fen}");
        let coup = moteur.meilleur_coup_fen(&fen, 30).expect("bestmove e.p.");
        let m = UciMove::from_ascii(coup.as_bytes())
            .expect("parse UCI")
            .to_move(&pos)
            .expect("coup légal");
        assert!(pos.legal_moves().contains(&m), "coup hors liste légale : {coup}");
    }

    /// Parsing des scores UCI sur transcriptions simulées, SANS moteur.
    /// Convention : score du point de vue du camp au trait, AUCUN renversement
    /// de signe — tanh(cp/300) tel quel, mate → ±1.
    #[test]
    fn parse_score_ligne_info() {
        // cp négatif : le trait est mal — transmis tel quel.
        assert_eq!(
            score_de_ligne_info("info depth 12 score cp -35 nodes 12345 pv e2e4"),
            Some((-35.0f32 / 300.0).tanh())
        );
        // mate négatif : le trait se fait mater → -1.
        assert_eq!(score_de_ligne_info("score mate -3"), Some(-1.0));
        // mate positif : le trait mate → +1 ; mate 0 : le trait EST maté → -1.
        assert_eq!(
            score_de_ligne_info("info depth 5 seldepth 5 score mate 2 pv h5f7"),
            Some(1.0)
        );
        assert_eq!(score_de_ligne_info("info depth 0 score mate 0"), Some(-1.0));
        // lowerbound/upperbound (après le nombre) : valeur bornée telle quelle
        // sur ce chemin d'étiquetage dans [-1, 1].
        assert_eq!(
            score_de_ligne_info("info depth 20 score cp 13 lowerbound nodes 99"),
            Some((13.0f32 / 300.0).tanh())
        );
        assert_eq!(
            score_de_ligne_info("info depth 20 score cp 250 upperbound"),
            Some((250.0f32 / 300.0).tanh())
        );
        // Lignes sans score exploitable → None (jamais de panique).
        assert_eq!(score_de_ligne_info("info string NNUE evaluation using nn.nnue"), None);
        assert_eq!(score_de_ligne_info("bestmove e2e4 ponder e7e5"), None);
        assert_eq!(score_de_ligne_info("info depth 1 score cp abc"), None);
        assert_eq!(score_de_ligne_info("info depth 1 score"), None);
    }

    /// Drapeau « borné » : c'est lui qui empêche une fenêtre d'aspiration
    /// ratée de se transformer en gaffe imaginaire au tableau de bord de
    /// l'arbitre (le score en CENTIPIONS, lui, n'a pas le droit d'être « au
    /// plus X »).
    #[test]
    fn parse_score_borne_ou_exact() {
        assert_eq!(
            score_brut_de_ligne_info("info depth 20 score cp 13 nodes 99 pv e2e4"),
            Some((ScoreUci::Cp(13), false))
        );
        assert_eq!(
            score_brut_de_ligne_info("info depth 20 score cp 13 lowerbound nodes 99"),
            Some((ScoreUci::Cp(13), true))
        );
        assert_eq!(
            score_brut_de_ligne_info("info depth 20 score cp 250 upperbound"),
            Some((ScoreUci::Cp(250), true))
        );
        assert_eq!(
            score_brut_de_ligne_info("info depth 7 score mate 3 upperbound"),
            Some((ScoreUci::Mat(3), true))
        );
        // « lowerbound » AVANT « score » (aucun moteur ne l'émet ainsi, mais la
        // recherche part de « score » : rien à confondre avec un coup nommé).
        assert_eq!(
            score_brut_de_ligne_info("info lowerbound depth 3 score cp 40"),
            Some((ScoreUci::Cp(40), false))
        );
        assert_eq!(score_brut_de_ligne_info("bestmove e2e4"), None);
    }

    /// Oracle pleine force réel : évaluations cohérentes, du point de vue du
    /// trait — position initiale proche de 0, camp au trait sans sa dame
    /// nettement négatif. `cargo test --lib -- --ignored evalue`.
    #[test]
    #[ignore = "nécessite engines/stockfish en local"]
    fn evalue_fen_pleine_force_reel() {
        let mut oracle =
            UciEngine::lance_pleine_force(CHEMIN, 30).expect("lancement de l'oracle");
        // Position initiale : équilibrée → |v| < 0.3.
        let v = oracle
            .evalue_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .expect("évaluation de la position initiale");
        assert!(v.abs() < 0.3, "position initiale : v = {v}");
        assert!(v.is_finite() && v.abs() <= 1.0);
        // Les NOIRS au trait, SANS leur dame (tout le reste symétrique) :
        // le trait est perdant → v < -0.5, sans renversement de signe.
        let v = oracle
            .evalue_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
            .expect("évaluation trait sans dame");
        assert!(v < -0.5, "trait sans dame : v = {v}");
        assert!(v.is_finite() && v >= -1.0);
    }

    /// Le StockfishBot complet joue un coup légal via le trait Bot.
    #[test]
    #[ignore = "nécessite engines/stockfish en local"]
    fn stockfish_bot_uci_joue_legal() {
        let mut bot = StockfishBot::new(CHEMIN, 1320, 30).expect("lancement du bot");
        let pos = Chess::default();
        let m = bot.choose(&pos).expect("un coup");
        assert!(pos.legal_moves().contains(&m));
    }

    /// ÉCHÉANCE : un processus VIVANT qui ne parlera jamais UCI (cmd.exe
    /// interactif — il bavarde puis attend stdin pour toujours) ne bloque pas
    /// `lance` : l'échéance du handshake tombe, le processus est tué et
    /// l'erreur est propre — là où un `read_line` direct aurait bloqué un
    /// ouvrier de self-play pour la nuit. Sans Stockfish, donc PAS ignoré.
    #[test]
    #[cfg(windows)]
    fn moteur_fige_tue_a_l_echeance() {
        let debut = Instant::now();
        let resultat = UciEngine::lance_avec_delai("cmd", Duration::from_millis(300));
        assert!(resultat.is_err(), "cmd.exe ne doit jamais réussir le handshake uci");
        assert!(
            debut.elapsed() < Duration::from_secs(8),
            "l'échéance devrait tomber en ~0,3 s, pas en {:?}",
            debut.elapsed()
        );
    }

    /// Un processus qui se termine sans jamais dire uciok (EOF immédiat) est
    /// détecté comme MORT : erreur propre et rapide, jamais de blocage ni de
    /// panique. Sans Stockfish, donc PAS ignoré.
    #[test]
    #[cfg(windows)]
    fn moteur_mort_eof_detecte() {
        let resultat = UciEngine::lance_avec_delai("whoami", Duration::from_secs(5));
        assert!(resultat.is_err(), "whoami n'est pas un moteur UCI");
    }
}
