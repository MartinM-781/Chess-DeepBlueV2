//! Ponder « préchauffage de table de transposition » : réfléchir sur le temps
//! de l'adversaire, sans machinerie de ponderhit.
//!
//! CONSTAT. Pendant le tour de l'adversaire, le moteur ne consomme rien. En
//! cadence lente (3 min/coup), c'est la moitié du temps de match jetée.
//!
//! MÉCANISME. Notre recherche livre à chaque itération sa VARIANTE PRINCIPALE
//! (`InfoIteration.pv`). Quand elle a choisi le coup C et que la PV valait
//! [C, R, ...], R est la RÉPONSE PRÉDITE de l'adversaire. On lance alors une
//! recherche de FOND sur P' = position après C puis R, sur un chercheur
//! DÉDIÉ mais posé sur la MÊME table de transposition que le champion
//! (`Recherche::jumeau_meme_tt`). Dès que l'adversaire joue son vrai coup, la
//! recherche de fond est rappelée (drapeau d'arrêt de
//! `Recherche::cherche_interruptible`) et JOINTE ; notre tour se déroule
//! ensuite normalement.
//!
//! CE QU'ON NE FAIT PAS. Le score de la recherche de fond n'est JAMAIS
//! réutilisé : à notre tour, on relance une recherche propre. Tout le gain
//! vient de la TT chaude — quand la prédiction est juste, l'arbre qu'on
//! s'apprête à explorer est déjà largement en table, donc profondeur
//! nettement supérieure à temps égal.
//!
//! CE QUE ÇA COÛTE QUAND LA PRÉDICTION EST FAUSSE. Pas une corruption : la
//! table est adressée par zobrist et validée par XOR (`search::CaseTT`), une
//! entrée étrangère ne peut jamais être LUE pour une autre position. Mais une
//! ÉVICTION, et celle-là est bien réelle. La politique de remplacement de
//! `Recherche::stocke` écrase inconditionnellement dès que la case porte une
//! AUTRE position (`cle_x ^ ancien != cle`) — sans regarder la profondeur de
//! l'ancienne entrée : une entrée de fond à profondeur 3 chasse une entrée du
//! champion à profondeur 20 tombée sur la même case. Or la TT SURVIT aux
//! coups d'une même partie, et les hits entre coups successifs sont « une
//! grosse part du gain » (doc de `Recherche::new`). Le ponder double donc le
//! brassage de la table, et la part correspondant aux prédictions fausses est
//! pure perte.
//!
//! DONC : le dispositif est SÛR, mais son gain net est un PARI, pas un acquis.
//! Il dépend du taux de prédiction — que le harnais publie précisément pour
//! qu'on puisse le mesurer. Ne pas activer --ponder en match classé sans un
//! A/B (protocole en tête de src/bin/match.rs). Deux repères pour cadrer
//! l'attente : le « +40 à +60 Elo » usuel du ponder vaut pour le VRAI ponder
//! (ponderhit — on réutilise la recherche elle-même) ; ici on ne transporte
//! que la table, le gain attendu est nettement plus faible et son SIGNE n'est
//! pas établi. Si l'A/B ne le fait pas sortir, la parade standard est un champ
//! d'âge/génération dans CaseTT, pour que la troisième clause de `stocke`
//! protège une entrée profonde du champion d'une entrée de fond superficielle.
//!
//! SÛRETÉ. Le thread de ponder ne survit jamais au coup adverse : `arrete()`
//! pose le drapeau PUIS joint, et l'implémentation de `Drop` refait la même
//! chose si l'objet part par un chemin imprévu (panique, sortie anticipée).
//! Le chercheur de fond n'a pas de hook `info` — il n'écrit donc jamais dans
//! le fichier du direct, qui reste la propriété exclusive de la boucle de
//! match.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use shakmaty::{Chess, Move, Position};

use crate::search::{Limites, Recherche};

/// Compteurs du ponder, cumulés sur tout le match : publiés dans
/// models/match_live.json et dans le récapitulatif final.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct StatsPonder {
    /// Recherches de fond effectivement DÉMARRÉES.
    pub lances: u64,
    /// Parmi elles, celles dont la réponse prédite était le vrai coup adverse.
    pub justes: u64,
    /// Temps de recherche de fond RÉELLEMENT passé, cumulé (ms). C'est le
    /// thread de fond qui chronomètre son propre appel de recherche, pas la
    /// boucle de match sa fenêtre d'attente : une recherche de fond qui se
    /// termine d'elle-même avant le coup adverse (mat prouvé dans l'horizon,
    /// PROF_MAX atteint) cesse d'être facturée à sa mort, pas à la jointure.
    pub ms_cumules: u64,
}

impl StatsPonder {
    /// Taux de prédictions justes dans [0, 1] (0 sans ponder lancé).
    pub fn taux(&self) -> f64 {
        if self.lances == 0 {
            0.0
        } else {
            self.justes as f64 / self.lances as f64
        }
    }
}

/// Réponse PRÉDITE de l'adversaire : le deuxième coup de la variante
/// principale de la recherche qui vient de choisir `joue`. None si la PV est
/// trop courte, ou si elle ne commence PAS par le coup effectivement joué —
/// une PV décorrélée du coup rendu (itération jetée, entrée de TT écrasée) ne
/// prédit rien, et mieux vaut ne pas ponderer que ponderer à côté.
pub fn reponse_predite(pv: &[Move], joue: &Move) -> Option<Move> {
    (pv.len() >= 2 && pv[0] == *joue).then(|| pv[1].clone())
}

/// Boîte aux lettres de la dernière variante principale livrée par le hook
/// d'information de la recherche. Le hook (thread principal du champion) y
/// dépose la PV de CHAQUE itération terminée ; la boucle de match relit la
/// dernière après coup. Mutex par principe — hook et boucle tournent en fait
/// sur le même thread, mais `Recherche::info` exige Send + Sync — et Arc pour
/// la capture 'static du hook.
#[derive(Clone, Default)]
pub struct DernierePv(Arc<Mutex<Vec<Move>>>);

impl DernierePv {
    pub fn nouvelle() -> Self {
        DernierePv::default()
    }

    /// Dépose la PV de l'itération qui vient de se terminer (appelé par le
    /// hook, hors chemin chaud). Un mutex empoisonné est ignoré : le direct
    /// et le ponder ne doivent jamais faire tomber le match.
    pub fn depose(&self, pv: &[Move]) {
        if let Ok(mut d) = self.0.lock() {
            d.clear();
            d.extend_from_slice(pv);
        }
    }

    /// Vide la boîte — à faire AVANT chaque recherche, pour ne jamais prédire
    /// à partir de la PV du coup précédent.
    pub fn vide(&self) {
        if let Ok(mut d) = self.0.lock() {
            d.clear();
        }
    }

    /// Réponse prédite après le coup `joue` (voir `reponse_predite`).
    pub fn reponse_apres(&self, joue: &Move) -> Option<Move> {
        let d = self.0.lock().ok()?;
        reponse_predite(&d, joue)
    }
}

/// Recherche de fond en vol : le thread, son drapeau d'arrêt, le coup prédit
/// et l'instant de départ.
struct EnCours {
    /// Le thread REND le chercheur — c'est ainsi que le ponder le récupère à
    /// la jointure, sans verrou ni partage d'état mutable — ET la durée de son
    /// appel de recherche, mesurée par lui-même. Cette durée-là est le vrai
    /// temps de fond : la fenêtre `debut..jointure`, elle, continue de courir
    /// après une fin naturelle de la recherche (mat prouvé, PROF_MAX).
    handle: std::thread::JoinHandle<(Recherche, Duration)>,
    arret: Arc<AtomicBool>,
    predit: Move,
    /// Départ de la FENÊTRE (spawn) : sert uniquement de repli quand le thread
    /// a paniqué et n'a donc rien pu rendre.
    debut: Instant,
}

/// Le ponder d'une partie : un chercheur de fond au repos entre deux coups,
/// au plus UNE recherche en vol à la fois, et les compteurs du match.
pub struct Ponder {
    /// Chercheur de fond au repos. None quand il est parti dans le thread
    /// (voir `encours`), quand le ponder est éteint (`--ponder` absent), ou
    /// après une panique du thread de fond (le ponder se désarme alors pour
    /// le reste du match plutôt que de faire tomber la partie).
    recherche: Option<Recherche>,
    encours: Option<EnCours>,
    stats: StatsPonder,
    /// Vrai si le ponder a été ARMÉ au lancement (`--ponder`). Indépendant
    /// d'un désarmement ultérieur (panique du thread de fond) : c'est ce
    /// drapeau, et lui seul, qui décide de publier le bloc de statistiques.
    arme: bool,
}

impl Ponder {
    /// Ponder ÉTEINT : `demarre` ne fait rien, `arrete` non plus, les
    /// compteurs restent à zéro. C'est le défaut du harnais de match.
    pub fn eteint() -> Self {
        Ponder { recherche: None, encours: None, stats: StatsPonder::default(), arme: false }
    }

    /// Ponder armé sur un chercheur de fond. Construire ce chercheur avec
    /// `Recherche::jumeau_meme_tt()` depuis la recherche du champion : c'est
    /// le PARTAGE DE LA TABLE qui fait tout l'intérêt du dispositif.
    pub fn arme(recherche: Recherche) -> Self {
        Ponder {
            recherche: Some(recherche),
            encours: None,
            stats: StatsPonder::default(),
            arme: true,
        }
    }

    pub fn stats(&self) -> StatsPonder {
        self.stats
    }

    /// Statistiques À PUBLIER : None quand le ponder n'a jamais été armé
    /// (option absente → champ JSON null, panneau web muet).
    pub fn stats_publiables(&self) -> Option<StatsPonder> {
        self.arme.then_some(self.stats)
    }

    /// Vrai si aucune recherche de fond n'est en vol (invariant attendu
    /// pendant notre propre recherche et entre deux parties).
    pub fn au_repos(&self) -> bool {
        self.encours.is_none()
    }

    /// Vrai si la recherche de fond a fini d'elle-même (elle a prouvé un mat
    /// dans l'horizon, ou épuisé la profondeur du moteur) — ou s'il n'y en a
    /// aucune. Ne joint pas : c'est une simple observation, utile aux tests
    /// pour attendre une fin NATURELLE plutôt qu'une durée de mur.
    pub fn fond_termine(&self) -> bool {
        self.encours.as_ref().is_none_or(|e| e.handle.is_finished())
    }

    /// Vrai si le ponder peut encore lancer des recherches de fond (armé et
    /// non désactivé par une panique).
    pub fn armable(&self) -> bool {
        self.recherche.is_some() || self.encours.is_some()
    }

    /// Démarre une recherche de fond sur P' = `apres_notre_coup` + `reponse`.
    /// No-op si le ponder est éteint, si une recherche est déjà en vol, si la
    /// réponse prédite n'est pas légale, si P' est terminale (rien à
    /// préchauffer) ou si P' est résolue par les tables Syzygy (la recherche
    /// sortirait aussitôt sans écrire une entrée : rien à préchauffer non
    /// plus). Ne bloque jamais.
    pub fn demarre(&mut self, apres_notre_coup: &Chess, reponse: &Move) {
        if self.encours.is_some() {
            return; // ceinture : jamais deux recherches de fond à la fois
        }
        let Some(mut r) = self.recherche.take() else { return };
        // Le coup prédit vient de la TT (marche de PV) : il est VALIDÉ ici
        // contre la position réelle, jamais joué sur parole.
        if !apres_notre_coup.is_legal(reponse) {
            self.recherche = Some(r);
            return;
        }
        let mut p = apres_notre_coup.clone();
        p.play_unchecked(reponse);
        // Rien à chercher : mat/pat prédit, ou racine lue dans les tables
        // (`cherche` rendrait le coup DTZ sans toucher à la TT). Dans les deux
        // cas on ne lance pas — et donc on ne COMPTE pas : un `lances` qui ne
        // préchauffe rien fausserait le taux comme le temps cumulé.
        if p.legal_moves().is_empty() || r.racine_resolue_par_syzygy(&p) {
            self.recherche = Some(r);
            return;
        }
        let arret = Arc::new(AtomicBool::new(false));
        let drapeau = arret.clone();
        let handle = std::thread::spawn(move || {
            // Le résultat est DÉLIBÉRÉMENT jeté : seul le préchauffage de la
            // table nous intéresse (voir l'en-tête du module). Le thread
            // chronomètre son PROPRE appel : c'est le seul endroit d'où l'on
            // voit la fin réelle de la recherche, y compris quand elle sort
            // d'elle-même bien avant le coup adverse.
            let t0 = Instant::now();
            let _ = r.cherche_interruptible(&p, Limites::illimitees(), drapeau);
            (r, t0.elapsed())
        });
        self.encours =
            Some(EnCours { handle, arret, predit: reponse.clone(), debut: Instant::now() });
        self.stats.lances += 1;
    }

    /// Rappelle la recherche de fond et la JOINT. À appeler dès l'arrivée du
    /// coup adverse, AVANT notre propre recherche. `coup_reel` = le coup
    /// effectivement joué (None si inconnu : forfait, fin de partie, arrêt de
    /// sûreté) — il ne sert qu'à créditer la prédiction. No-op si rien n'est
    /// en vol.
    pub fn arrete(&mut self, coup_reel: Option<&Move>) {
        let Some(en_cours) = self.encours.take() else { return };
        en_cours.arret.store(true, Ordering::Relaxed);
        // Le temps crédité est celui que le THREAD a mesuré autour de son
        // propre appel : la fenêtre `debut..ici` la surestimerait de tout
        // l'écart entre une fin naturelle de la recherche (mat prouvé dans
        // l'horizon, PROF_MAX atteint) et l'arrivée du coup adverse — en
        // finale, cet écart peut valoir le movetime entier de l'adversaire.
        let duree = match en_cours.handle.join() {
            // Le chercheur revient au repos, prêt pour le coup suivant.
            Ok((r, duree)) => {
                self.recherche = Some(r);
                duree
            }
            Err(_) => {
                eprintln!(
                    "ponder : le thread de fond a paniqué — ponder désarmé pour la suite du match"
                );
                en_cours.debut.elapsed() // repli : le thread n'a rien rendu
            }
        };
        self.stats.ms_cumules += duree.as_millis() as u64;
        if coup_reel.is_some_and(|m| *m == en_cours.predit) {
            self.stats.justes += 1;
        }
    }

    /// À appeler entre deux PARTIES, à côté de `Recherche::nouvelle_partie()`
    /// du champion. Joint un éventuel fond en vol, puis remet à zéro les
    /// heuristiques de tri du chercheur de fond (killers, historique).
    ///
    /// Pourquoi : `Recherche::nouvelle_partie()` n'est appelée que sur le
    /// CHAMPION. Elle vide la table — partagée, donc vidée aussi pour le fond
    /// — mais les killers et l'historique sont PAR CHERCHEUR : sans ce rappel,
    /// le chercheur de fond traverse tout le match avec des heuristiques
    /// héritées de parties qui n'existent plus. L'effet sur la force est nul
    /// (ces heuristiques ne font que trier, et le score du fond n'est jamais
    /// réutilisé) ; ce qui compte, c'est que son comportement redevienne
    /// reproductible à partir du seul état de la partie.
    ///
    /// La table N'EST PAS re-vidée ici : c'est l'affaire du champion, seul
    /// propriétaire du calendrier des parties.
    ///
    /// Les compteurs `stats` sont CUMULÉS sur tout le match et survivent donc
    /// (c'est ce que publient le direct et le récapitulatif).
    pub fn nouvelle_partie(&mut self) {
        self.arrete(None);
        if let Some(r) = self.recherche.as_mut() {
            r.oublie_heuristiques();
        }
    }
}

impl Drop for Ponder {
    /// Ceinture : aucun thread de ponder ne survit à l'objet, même si la
    /// boucle de match sort par un chemin imprévu.
    fn drop(&mut self) {
        self.arrete(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::N_FEATURES;
    use crate::nn::Mlp;
    use crate::search::SCORE_MAT;
    use shakmaty::fen::Fen;
    use shakmaty::CastlingMode;

    /// Réseau [773 → 12 → 1] : évaluation non triviale et déterministe, assez
    /// petite pour que ces tests tournent en profil dev (même motivation que
    /// le `reseau_reduit` des tests de search.rs).
    fn petit_reseau() -> Arc<Mlp> {
        Arc::new(Mlp::new_avec_tailles(&[N_FEATURES, 12, 1], 7))
    }

    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    fn coup_uci(pos: &Chess, uci: &str) -> Move {
        pos.legal_moves()
            .into_iter()
            .find(|m| m.to_uci(CastlingMode::Standard).to_string() == uci)
            .unwrap_or_else(|| panic!("coup {uci} illégal ici"))
    }

    fn ponder_arme(threads: u32) -> Ponder {
        let mut champion = Recherche::new(petit_reseau(), 16);
        champion.threads = threads;
        Ponder::arme(champion.jumeau_meme_tt())
    }

    /// Prédiction : la réponse est le 2e coup de la PV, et seulement si la PV
    /// commence par le coup effectivement joué.
    #[test]
    fn reponse_predite_exige_une_pv_alignee() {
        let depart = Chess::default();
        let e4 = coup_uci(&depart, "e2e4");
        let d4 = coup_uci(&depart, "d2d4");
        let apres_e4 = depart.clone().play(&e4).expect("coup légal");
        let e5 = coup_uci(&apres_e4, "e7e5");

        assert_eq!(reponse_predite(&[e4.clone(), e5.clone()], &e4), Some(e5.clone()));
        // PV trop courte : rien à prédire.
        assert_eq!(reponse_predite(&[e4.clone()], &e4), None);
        assert_eq!(reponse_predite(&[], &e4), None);
        // PV décorrélée du coup joué : on refuse de prédire.
        assert_eq!(reponse_predite(&[e4.clone(), e5.clone()], &d4), None);

        // Même contrat au travers de la boîte aux lettres, vidage compris.
        let boite = DernierePv::nouvelle();
        assert_eq!(boite.reponse_apres(&e4), None);
        boite.depose(&[e4.clone(), e5.clone()]);
        assert_eq!(boite.reponse_apres(&e4), Some(e5));
        assert_eq!(boite.reponse_apres(&d4), None);
        boite.vide();
        assert_eq!(boite.reponse_apres(&e4), None);
    }

    /// Ponder ÉTEINT (défaut du harnais) : rien ne démarre, rien ne se joint,
    /// les compteurs restent à zéro.
    #[test]
    fn ponder_eteint_ne_fait_rien() {
        let mut p = Ponder::eteint();
        let depart = Chess::default();
        let e4 = coup_uci(&depart, "e2e4");
        let apres = depart.clone().play(&e4).expect("coup légal");
        let e5 = coup_uci(&apres, "e7e5");
        p.demarre(&apres, &e5);
        assert!(p.au_repos());
        assert!(!p.armable());
        p.arrete(Some(&e5));
        assert_eq!(p.stats(), StatsPonder::default());
    }

    /// LE test du contrat : 3 coups simulés avec ponder (mono-thread et SMP).
    /// À chaque coup on démarre une recherche de fond puis on l'arrête avec
    /// le vrai coup adverse — juste, puis fausse, puis juste. On vérifie :
    /// la JOINTURE (le chercheur revient toujours au repos, donc le thread
    /// est provablement terminé — `join` ne rend la main qu'à sa mort — et il
    /// n'a pas paniqué puisqu'il a rendu le chercheur), l'absence de thread
    /// résiduel entre deux coups, et la cohérence des compteurs.
    #[test]
    fn ponder_trois_coups_jointure_et_compteurs() {
        for threads in [1u32, 4] {
            let mut p = ponder_arme(threads);
            let mut pos = Chess::default();
            // (coup du champion, réponse prédite, vrai coup adverse)
            let scenario = [
                ("e2e4", "e7e5", "e7e5"), // prédiction juste
                ("g1f3", "b8c6", "g8f6"), // prédiction fausse
                ("f1c4", "f6e4", "f6e4"), // prédiction juste
            ];
            let mut justes_attendus = 0u64;
            for (i, (notre, predit, reel)) in scenario.iter().enumerate() {
                // Notre coup est joué : la position de ponder est P'.
                let notre = coup_uci(&pos, notre);
                pos.play_unchecked(&notre);
                let predit = coup_uci(&pos, predit);
                let reel = coup_uci(&pos, reel);

                assert!(p.au_repos(), "thread résiduel avant le coup {i}");
                p.demarre(&pos, &predit);
                assert!(!p.au_repos(), "la recherche de fond n'a pas démarré (coup {i})");
                // Le « temps de réflexion de l'adversaire », en miniature.
                std::thread::sleep(std::time::Duration::from_millis(30));
                p.arrete(Some(&reel));

                assert!(p.au_repos(), "thread non joint après le coup {i}");
                assert!(p.armable(), "chercheur de fond non récupéré (coup {i})");
                if predit == reel {
                    justes_attendus += 1;
                }
                assert_eq!(p.stats().lances, i as u64 + 1);
                assert_eq!(p.stats().justes, justes_attendus);
                pos.play_unchecked(&reel);
            }
            let s = p.stats();
            assert_eq!((s.lances, s.justes), (3, 2), "compteurs ({threads} thread(s))");
            assert!((s.taux() - 2.0 / 3.0).abs() < 1e-9);
            // Trois ponders d'au moins 30 ms : le temps cumulé est crédité.
            assert!(s.ms_cumules >= 60, "temps de ponder cumulé invraisemblable : {s:?}");
        }
    }

    /// LE test du gain : préchauffage bout en bout. La recherche de FOND part
    /// sur la position prédite pendant « le tour de l'adversaire », puis le
    /// champion cherche CETTE MÊME position, au MÊME budget de nœuds, sur la
    /// table PARTAGÉE. Table froide, il ne voit rien ; table préchauffée, il
    /// sort le mat forcé en une poignée de nœuds — c'est exactement le gain
    /// visé (le score de la recherche de fond, lui, n'est jamais réutilisé :
    /// tout passe par la table).
    ///
    /// DÉTERMINISTE et insensible à la charge de la machine : la position
    /// prédite porte un mat en 2, la recherche de fond s'arrête donc D'ELLE-
    /// MÊME (mat prouvé dans l'horizon) et on l'attend par `fond_termine()`,
    /// jamais par une durée de mur.
    #[test]
    fn ponder_prechauffe_bien_la_table_du_champion() {
        // Échelle de tours : 1.Ta2 (notre coup) Rh8 (réponse prédite), et la
        // position qui suit est un mat en 2 pour les blancs.
        let avant = pos_de_fen("6k1/8/8/8/8/R7/8/1R5K w - - 0 1");
        let notre = coup_uci(&avant, "a3a2");
        let mut apres = avant.clone();
        apres.play_unchecked(&notre);
        let predit = coup_uci(&apres, "g8h8");
        let mut cible = apres.clone();
        cible.play_unchecked(&predit);
        let limites = Limites { max_noeuds: 600, max_profondeur: 0, movetime_ms: 0 };

        // Témoin : le champion cherche la cible à table FROIDE — à ce budget,
        // il épuise ses nœuds sans jamais voir le mat.
        let mut froid = Recherche::new(petit_reseau(), 16);
        let a = froid.cherche(&cible, limites);
        assert!(a.score < 900.0, "témoin invalide : le mat est vu à table froide ({})", a.score);

        // Avec ponder : la recherche de fond tourne sur la position prédite
        // et s'arrête d'elle-même en prouvant le mat.
        let mut champion = Recherche::new(petit_reseau(), 16);
        let mut p = Ponder::arme(champion.jumeau_meme_tt());
        p.demarre(&apres, &predit);
        let t0 = Instant::now();
        while !p.fond_termine() && t0.elapsed().as_secs() < 60 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(p.fond_termine(), "la recherche de fond n'a pas abouti au mat");
        p.arrete(Some(&predit));
        assert!(p.au_repos() && p.armable(), "chercheur de fond non repris");

        // Le champion, sur la MÊME table : le mat en 2 (SCORE_MAT - 3) sort
        // en une fraction des nœuds du témoin.
        let b = champion.cherche(&cible, limites);
        let coup = b.coup.expect("coup légal");
        assert!(cible.is_legal(&coup));
        assert!(
            (b.score - (SCORE_MAT - 3.0)).abs() < 1e-3,
            "table non préchauffée : score {} au lieu du mat en 2 ({})",
            b.score,
            SCORE_MAT - 3.0
        );
        assert!(
            b.noeuds < a.noeuds,
            "arbre non réduit : {} nœuds contre {} à table froide",
            b.noeuds,
            a.noeuds
        );
        let s = p.stats();
        assert_eq!((s.lances, s.justes), (1, 1));
    }

    /// LA BRANCHE DÉFAVORABLE — prédiction FAUSSE — côté sûreté. La recherche
    /// de fond a copieusement écrit dans la table pour une position SANS
    /// RAPPORT ; le champion cherche ensuite la vraie position, au même budget
    /// qu'à table vierge. La validation XOR rend toute entrée étrangère
    /// invisible à la sonde : l'arbre exploré est le même AU NŒUD PRÈS, et le
    /// résultat identique AU BIT. Autrement dit une prédiction ratée ne peut
    /// ni dévier ni élargir la recherche du champion.
    ///
    /// CE QUE CE TEST NE COUVRE PAS, et qu'aucun test unitaire ne peut
    /// couvrir : le coût d'ÉVICTION. Il n'apparaît que sur une table DÉJÀ
    /// chargée par les coups précédents d'une vraie partie — la recherche de
    /// fond y chasse des entrées profondes du champion (voir l'en-tête du
    /// module). Ce coût-là se mesure par l'A/B --ponder on/off décrit en tête
    /// de src/bin/match.rs, pas ici.
    #[test]
    fn ponder_prediction_fausse_ne_degrade_pas_la_recherche() {
        // Deux positions entre lesquelles aucune transposition n'est possible
        // (matériels incompatibles) : ce que le fond écrit pour l'une ne peut
        // JAMAIS être lu pour l'autre — seules restent les collisions
        // d'index, et c'est précisément ce qu'on veut éprouver.
        let cible =
            pos_de_fen("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
        let fausse = pos_de_fen("8/5k2/8/8/8/8/4PPP1/4K2R w - - 0 1");
        let coup_fausse = coup_uci(&fausse, "h1h5");
        let limites = Limites { max_noeuds: 2000, max_profondeur: 0, movetime_ms: 0 };

        // Témoin : table VIERGE.
        let mut froid = Recherche::new(petit_reseau(), 16);
        let a = froid.cherche(&cible, limites);

        // Avec un ponder qui s'est trompé de position.
        let champion_init = Recherche::new(petit_reseau(), 16);
        let mut p = Ponder::arme(champion_init.jumeau_meme_tt());
        let mut champion = champion_init;
        p.demarre(&fausse, &coup_fausse);
        assert!(!p.au_repos(), "la recherche de fond n'a pas démarré");
        std::thread::sleep(std::time::Duration::from_millis(200));
        p.arrete(None); // le vrai coup adverse n'était pas celui prédit
        assert!(p.au_repos() && p.armable(), "chercheur de fond non repris");
        assert_eq!(p.stats().justes, 0, "prédiction ratée créditée comme juste");

        let b = champion.cherche(&cible, limites);
        assert_eq!(a.coup, b.coup, "coup dévié par une prédiction fausse");
        assert_eq!(a.score.to_bits(), b.score.to_bits(), "score changé (au bit)");
        assert_eq!(a.profondeur, b.profondeur, "profondeur perdue");
        assert_eq!(a.noeuds, b.noeuds, "arbre élargi par une prédiction fausse");
    }

    /// `ms_cumules` compte le TRAVAIL, pas la fenêtre d'attente. La recherche
    /// de fond part sur une position qui porte un mat en 2 : elle le prouve et
    /// s'arrête D'ELLE-MÊME, longtemps avant le « coup adverse ». Le temps
    /// crédité doit s'arrêter avec elle — sans quoi une finale afficherait le
    /// movetime entier de l'adversaire comme du temps de réflexion.
    #[test]
    fn ms_cumules_ne_facture_pas_l_attente_apres_une_fin_naturelle() {
        let avant = pos_de_fen("6k1/8/8/8/8/R7/8/1R5K w - - 0 1");
        let notre = coup_uci(&avant, "a3a2");
        let mut apres = avant.clone();
        apres.play_unchecked(&notre);
        let predit = coup_uci(&apres, "g8h8");

        let champion = Recherche::new(petit_reseau(), 16);
        let mut p = Ponder::arme(champion.jumeau_meme_tt());
        let fenetre = Instant::now();
        p.demarre(&apres, &predit);
        let t0 = Instant::now();
        while !p.fond_termine() && t0.elapsed().as_secs() < 60 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(p.fond_termine(), "la recherche de fond n'a pas abouti au mat");
        // « L'adversaire réfléchit encore » : le thread, lui, est déjà mort.
        std::thread::sleep(std::time::Duration::from_millis(600));
        p.arrete(Some(&predit));
        let fenetre_ms = fenetre.elapsed().as_millis() as u64;
        let s = p.stats();
        assert_eq!(s.lances, 1);
        assert!(
            s.ms_cumules + 400 <= fenetre_ms,
            "attente facturée comme réflexion : {} ms crédités pour une fenêtre de {} ms",
            s.ms_cumules,
            fenetre_ms
        );
    }

    /// Entre deux parties : le chercheur de fond doit oublier ses heuristiques
    /// de tri comme le champion oublie les siennes. `Ponder::nouvelle_partie`
    /// joint un éventuel fond en vol, remet killers et historique à zéro, et
    /// PRÉSERVE les compteurs (cumulés sur tout le match).
    #[test]
    fn ponder_nouvelle_partie_joint_et_preserve_les_compteurs() {
        let depart = Chess::default();
        let e4 = coup_uci(&depart, "e2e4");
        let mut apres = depart.clone();
        apres.play_unchecked(&e4);
        let e5 = coup_uci(&apres, "e7e5");

        let mut p = ponder_arme(1);
        p.demarre(&apres, &e5);
        std::thread::sleep(std::time::Duration::from_millis(20));
        p.arrete(Some(&e5));
        let avant = p.stats();
        assert_eq!((avant.lances, avant.justes), (1, 1));

        // Fond EN VOL au changement de partie : nouvelle_partie doit joindre.
        p.demarre(&apres, &e5);
        assert!(!p.au_repos());
        p.nouvelle_partie();
        assert!(p.au_repos(), "fond non joint par nouvelle_partie");
        assert!(p.armable(), "chercheur de fond non repris");
        // Compteurs cumulés : le second départ compte, rien n'est remis à zéro.
        let apres_stats = p.stats();
        assert_eq!(apres_stats.lances, 2);
        assert_eq!(apres_stats.justes, avant.justes, "prédiction créditée sans coup réel");
    }

    /// Drop : un ponder détruit avec une recherche EN VOL la rappelle et la
    /// joint (aucun thread ne survit à l'objet). Le test échouerait par
    /// blocage si la jointure manquait.
    #[test]
    fn ponder_drop_joint_la_recherche_en_vol() {
        let depart = Chess::default();
        let e4 = coup_uci(&depart, "e2e4");
        let mut apres = depart.clone();
        apres.play_unchecked(&e4);
        let e5 = coup_uci(&apres, "e7e5");

        let mut p = ponder_arme(2);
        p.demarre(&apres, &e5);
        assert!(!p.au_repos());
        drop(p); // doit poser le drapeau et joindre, sans traîner
    }
}
