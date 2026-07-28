//! Le « micro » du direct — retransmission d'UNE partie de self-play à la fois.
//!
//! L'entraîneur joue de nombreuses parties en parallèle (rayon) ; une seule
//! « prend le micro » et publie sa progression coup par coup dans
//! models/live.json (servi tel quel par serve.exe sur GET /api/live, affiché
//! par la page /live). Conçu pour un coût NUL quand personne ne publie : les
//! parties sans micro ne paient qu'UN compare_exchange raté au démarrage.
//!
//! Contrat JSON publié (voir la page web/live.html) :
//! {"actif": bool, "cycle": u64, "ply": u32, "fen": "...",
//!  "last_move": "uci"|null, "history_san": ["e4", ...],
//!  "v_eleve": f32|null, "v_prof": f32|null,
//!  "phase": "ouverture"|"normale", "result": null|"1-0"|"0-1"|"1/2-1/2",
//!  "resultat_precedent": null|"1-0"|"0-1"|"1/2-1/2",
//!  "depart": "etiquette"|null}
//! Les v_* sont CÔTÉ BLANCS (la conversion depuis la perspective du trait est
//! faite par l'appelant, dans selfplay.rs) ; v_prof est null hors mentorat ;
//! result est non-null uniquement sur la publication finale d'une partie ;
//! depart est l'étiquette de la position de départ tirée par departs::tirage
//! (« ouverture:najdorf », « finale:KRPvKR », « initiale »), null pour une
//! partie des variantes historiques — un front qui ignore cette clé n'affiche
//! simplement rien (clé supplémentaire sans effet, web/live.js vérifié).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// Le micro : false = libre, true = tenu par un `Journaliste` vivant.
static MICRO: AtomicBool = AtomicBool::new(false);
/// Chemin du JSON publié — configuré UNE fois au démarrage du process
/// (train.rs) ; jamais configuré dans serve.exe ou les tests d'autres modules,
/// où `prendre_le_micro` renvoie alors None sans rien coûter.
static CHEMIN: OnceLock<String> = OnceLock::new();
/// Résultat de la dernière partie publiée TERMINÉE (« resultat_precedent »
/// du contrat : la page /live peut afficher l'issue de la partie d'avant).
static RESULTAT_PRECEDENT: Mutex<Option<String>> = Mutex::new(None);
/// Cycle d'entraînement en cours, annoncé par train.rs à chaque cycle
/// (les parties de self-play ne connaissent pas le cycle : elles le lisent ici).
static CYCLE: AtomicU64 = AtomicU64::new(0);

/// Configure le chemin du fichier publié (ex. "models/live.json"). À appeler
/// une fois au démarrage de l'entraîneur ; les appels suivants sont ignorés.
pub fn configure(chemin_json: &str) {
    let _ = CHEMIN.set(chemin_json.to_string());
}

/// Annonce le numéro du cycle en cours (affiché par la page /live).
pub fn annonce_cycle(cycle: u64) {
    CYCLE.store(cycle, Ordering::Relaxed);
}

/// Dernier cycle annoncé (0 tant que train.rs n'a rien annoncé).
pub fn cycle_courant() -> u64 {
    CYCLE.load(Ordering::Relaxed)
}

/// Détenteur du micro (RAII) : tant qu'il vit, aucune autre partie ne peut
/// publier ; son Drop rend le micro. Ne se construit que via
/// `prendre_le_micro`.
pub struct Journaliste {
    _prive: (),
}

impl Drop for Journaliste {
    fn drop(&mut self) {
        MICRO.store(false, Ordering::Release);
    }
}

/// Tente de prendre le micro : None si personne n'a configuré le direct
/// (process sans dashboard, tests) ou si une autre partie le tient déjà.
/// C'est le SEUL coût payé par les parties non retransmises.
pub fn prendre_le_micro() -> Option<Journaliste> {
    CHEMIN.get()?;
    MICRO
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .ok()
        .map(|_| Journaliste { _prive: () })
}

impl Journaliste {
    /// Publie l'état courant de la partie retransmise : sérialise le contrat
    /// JSON et l'écrit ATOMIQUEMENT (.tmp puis rename, comme copie_atomique
    /// de train.rs — un lecteur ne voit jamais de fichier partiel). Les v_*
    /// doivent déjà être CÔTÉ BLANCS. `result` non-null = publication finale :
    /// il est alors mémorisé pour le « resultat_precedent » des publications
    /// suivantes. Les erreurs d'E/S sont ignorées : le direct ne doit jamais
    /// faire tomber l'entraînement.
    #[allow(clippy::too_many_arguments)]
    pub fn publie(
        &self,
        cycle: u64,
        ply: u32,
        fen: &str,
        last_move: Option<&str>,
        history_san: &[String],
        v_eleve: Option<f32>,
        v_prof: Option<f32>,
        phase: &str,
        result: Option<&str>,
    ) {
        // Signature historique INTACTE : délègue avec depart = None (la clé
        // « depart » est alors publiée à null, comme pour toute partie des
        // variantes historiques).
        self.publie_avec_depart(
            cycle, ply, fen, last_move, history_san, v_eleve, v_prof, phase, result, None,
        );
    }

    /// Comme `publie`, avec en plus l'étiquette du DÉPART de la partie
    /// retransmise sous la clé « depart » du contrat : « ouverture:najdorf »,
    /// « finale:KRPvKR », « initiale »… (voir src/departs.rs) — la page /live
    /// peut ainsi indiquer la provenance d'une partie qui démarre d'une
    /// position à 4-5 pièces sans historique SAN. None = variante historique
    /// (clé publiée à null).
    #[allow(clippy::too_many_arguments)]
    pub fn publie_avec_depart(
        &self,
        cycle: u64,
        ply: u32,
        fen: &str,
        last_move: Option<&str>,
        history_san: &[String],
        v_eleve: Option<f32>,
        v_prof: Option<f32>,
        phase: &str,
        result: Option<&str>,
        depart: Option<&str>,
    ) {
        let Some(chemin) = CHEMIN.get() else { return };
        // « resultat_precedent » : l'issue de la partie publiée d'AVANT ;
        // si celle-ci se termine (result non-null), elle devient à son tour
        // le précédent des publications futures.
        let precedent = {
            let mut garde = RESULTAT_PRECEDENT.lock().unwrap();
            let avant = garde.clone();
            if let Some(r) = result {
                *garde = Some(r.to_string());
            }
            avant
        };
        let v = serde_json::json!({
            "actif": true,
            "cycle": cycle,
            "ply": ply,
            "fen": fen,
            "last_move": last_move,
            "history_san": history_san,
            "v_eleve": v_eleve,
            "v_prof": v_prof,
            "phase": phase,
            "result": result,
            "resultat_precedent": precedent,
            "depart": depart,
        });
        let tmp = format!("{chemin}.tmp");
        if std::fs::write(&tmp, v.to_string()).is_ok() {
            let _ = std::fs::rename(&tmp, chemin);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    /// Configure le direct vers un fichier temporaire et renvoie le chemin
    /// RÉELLEMENT configuré (CHEMIN est un OnceLock global au binaire de
    /// test : seul le premier configure gagne, les deux tests partagent).
    fn configure_test() -> &'static str {
        let chemin = std::env::temp_dir()
            .join(format!("echec_live_test_{}.json", std::process::id()));
        configure(&chemin.to_string_lossy());
        CHEMIN.get().expect("chemin configuré").as_str()
    }

    /// Prend le micro avec patience : d'autres tests du binaire (self-play)
    /// peuvent le tenir le temps d'une partie — on attend qu'il se libère.
    fn prendre_avec_patience() -> Journaliste {
        loop {
            if let Some(j) = prendre_le_micro() {
                return j;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Micro exclusif : tant qu'un Journaliste vit, un second thread n'obtient
    /// rien ; après le Drop, le micro redevient disponible.
    #[test]
    fn micro_exclusif_puis_libere_au_drop() {
        configure_test();
        let j = prendre_avec_patience();
        let barriere = Barrier::new(2);
        std::thread::scope(|s| {
            s.spawn(|| {
                barriere.wait();
                // Le micro est tenu par `j` : l'autre thread perd, garanti.
                assert!(
                    prendre_le_micro().is_none(),
                    "le micro tenu ne doit pas être pris par un second thread"
                );
            });
            barriere.wait();
        });
        drop(j);
        // Après le Drop, on récupère le micro.
        let j2 = prendre_avec_patience();
        drop(j2);
    }

    /// publie() écrit un JSON parsable conforme au contrat, et le résultat
    /// publié devient le « resultat_precedent » de la publication suivante.
    #[test]
    fn publie_json_conforme_au_contrat() {
        let chemin = configure_test();
        let j = prendre_avec_patience();

        // Publication finale d'une partie (résultat 1-0), sans mentor.
        j.publie(
            3,
            12,
            "8/8/8/8/8/8/8/K6k w - - 0 1",
            Some("e2e4"),
            &["e4".to_string()],
            Some(0.25),
            None,
            "normale",
            Some("1-0"),
        );
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(chemin).unwrap()).unwrap();
        assert_eq!(v["actif"], true);
        assert_eq!(v["cycle"], 3);
        assert_eq!(v["ply"], 12);
        assert_eq!(v["fen"], "8/8/8/8/8/8/8/K6k w - - 0 1");
        assert_eq!(v["last_move"], "e2e4");
        assert_eq!(v["history_san"], serde_json::json!(["e4"]));
        assert!((v["v_eleve"].as_f64().unwrap() - 0.25).abs() < 1e-6);
        assert!(v["v_prof"].is_null(), "v_prof null hors mentorat");
        assert_eq!(v["phase"], "normale");
        assert_eq!(v["result"], "1-0");
        assert!(v["depart"].is_null(), "depart null via publie (signature historique)");

        // Publication suivante (nouvelle partie, mode mentoré) : le résultat
        // d'avant est devenu « resultat_precedent ».
        j.publie(4, 1, "fen2", None, &[], None, Some(-0.5), "ouverture", None);
        let v2: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(chemin).unwrap()).unwrap();
        assert!(v2["result"].is_null());
        assert_eq!(v2["resultat_precedent"], "1-0");
        assert!(v2["last_move"].is_null());
        assert!(v2["v_eleve"].is_null());
        assert!((v2["v_prof"].as_f64().unwrap() + 0.5).abs() < 1e-6);
        assert_eq!(v2["phase"], "ouverture");
        assert_eq!(v2["history_san"], serde_json::json!([]));

        // Publication via publie_avec_depart : l'étiquette du départ est
        // retransmise telle quelle sous la clé « depart » (le reste du
        // contrat est inchangé — resultat_precedent survit).
        j.publie_avec_depart(4, 2, "fen3", None, &[], None, None, "normale", None,
                             Some("finale:KRPvKR"));
        let v3: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(chemin).unwrap()).unwrap();
        assert_eq!(v3["depart"], "finale:KRPvKR");
        assert_eq!(v3["resultat_precedent"], "1-0");
    }
}
