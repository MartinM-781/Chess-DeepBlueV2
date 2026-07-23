//! Client UCI minimal pour piloter un moteur externe (Stockfish) à force
//! limitée (UCI_LimitStrength + UCI_Elo), afin de recaler l'échelle Elo maison
//! sur une référence réelle (voir src/bin/calibrate.rs).
//!
//! Points de vigilance couverts ici :
//! - lecture ligne à ligne bloquante UNIQUEMENT quand une réponse est attendue
//!   (uciok, readyok, bestmove) : les `info ...` émis pendant `go` sont drainés
//!   dans la même boucle, donc pas de deadlock tuyau plein ;
//! - stderr redirigé vers null (un moteur bavard ne peut pas se bloquer dessus) ;
//! - EOF détecté (moteur mort → erreur propre, jamais de boucle infinie) ;
//! - bornes UCI_Elo lues dans la sortie de `uci` et clamp systématique ;
//! - coups de promotion à 5 caractères acceptés (parse via shakmaty::uci) ;
//! - FEN générée avec EnPassantMode::Legal (case e.p. présente seulement si
//!   une prise en passant est réellement légale) ;
//! - Drop : `quit` poli, attente courte, puis kill — pas de processus zombie.

use std::io::{BufRead, BufReader, Error, ErrorKind, Result, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
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

/// Moteur UCI externe (processus enfant + tuyaux).
pub struct UciEngine {
    enfant: Child,
    entree: ChildStdin,
    sortie: BufReader<ChildStdout>,
    /// Bornes du spin UCI_Elo annoncées par le moteur.
    pub elo_min: u32,
    pub elo_max: u32,
}

/// Erreur d'E/S étiquetée (contexte du protocole UCI).
fn erreur(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

impl UciEngine {
    /// Lance le moteur et fait le handshake `uci` → `uciok`, en relevant les
    /// bornes du spin UCI_Elo au passage.
    pub fn lance(chemin: &str) -> Result<UciEngine> {
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
        let sortie = BufReader::new(
            enfant
                .stdout
                .take()
                .ok_or_else(|| erreur("stdout du moteur indisponible".into()))?,
        );
        let mut moteur = UciEngine {
            enfant,
            entree,
            sortie,
            elo_min: ELO_MIN_DEFAUT,
            elo_max: ELO_MAX_DEFAUT,
        };
        moteur.envoie("uci")?;
        // Draine l'en-tête (id, option ...) jusqu'à uciok, en relevant les
        // bornes de « option name UCI_Elo type spin default X min Y max Z ».
        loop {
            let ligne = moteur.ligne()?;
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

    /// Lit une ligne complète ; EOF (moteur mort) → erreur propre.
    fn ligne(&mut self) -> Result<String> {
        let mut tampon = String::new();
        let n = self.sortie.read_line(&mut tampon)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "le moteur UCI a fermé sa sortie (processus mort ?)",
            ));
        }
        Ok(tampon)
    }

    /// `isready` → attend `readyok` (draine tout le reste, y compris d'éventuels
    /// `info` tardifs).
    pub fn pret(&mut self) -> Result<()> {
        self.envoie("isready")?;
        loop {
            if self.ligne()?.trim() == "readyok" {
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

    /// Demande le meilleur coup sur une FEN donnée en `movetime_ms` ms.
    /// Renvoie le coup UCI brut (« e2e4 », promotion « e7e8q » à 5 caractères).
    /// Les lignes `info ...` émises pendant la recherche sont consommées ici
    /// même — c'est ce qui évite le deadlock « le moteur écrit, personne ne lit ».
    pub fn meilleur_coup_fen(&mut self, fen: &str, movetime_ms: u64) -> Result<String> {
        self.envoie(&format!("position fen {fen}"))?;
        self.envoie(&format!("go movetime {movetime_ms}"))?;
        loop {
            let ligne = self.ligne()?;
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
                return Ok(coup.to_string());
            }
            // Sinon : ligne info/string, on continue à drainer.
        }
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        // Sortie polie ; si le moteur traîne (ou si l'écriture échoue), kill.
        let _ = self.envoie("quit");
        let debut = Instant::now();
        loop {
            match self.enfant.try_wait() {
                Ok(Some(_)) => return, // terminé proprement, pas de zombie
                Ok(None) if debut.elapsed() < DELAI_QUIT => {
                    std::thread::sleep(Duration::from_millis(10));
                }
                _ => break,
            }
        }
        let _ = self.enfant.kill();
        let _ = self.enfant.wait();
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

    /// Le StockfishBot complet joue un coup légal via le trait Bot.
    #[test]
    #[ignore = "nécessite engines/stockfish en local"]
    fn stockfish_bot_uci_joue_legal() {
        let mut bot = StockfishBot::new(CHEMIN, 1320, 30).expect("lancement du bot");
        let pos = Chess::default();
        let m = bot.choose(&pos).expect("un coup");
        assert!(pos.legal_moves().contains(&m));
    }
}
