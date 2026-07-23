//! Recherche « sérieuse » : négamax alpha-bêta à approfondissement itératif,
//! table de transposition, quiescence, tri des coups (coup TT, MVV-LVA,
//! killers, historique), élagage null-move. Les feuilles calmes sont évaluées
//! par le réseau (perspective du trait, [-1,1]) ; les mats sont exacts.
//!
//! C'est l'étage 1 de la fusée « battre Deep Blue » : il sert à la fois à
//! JOUER (serveur, arène) et à FABRIQUER les étiquettes TD-leaf du self-play
//! (le score racine devient la cible d'apprentissage).
//!
//! Échelle des scores : réseau dans [-1, 1] ; mats à ±(SCORE_MAT - ply) pour
//! préférer les mats courts — SCORE_MAT domine largement l'échelle réseau.

use std::cmp::Reverse;
use std::sync::Arc;
use std::time::{Duration, Instant};

use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Move, Position};

use crate::features::{encode, N_FEATURES};
use crate::nn::Mlp;

pub const SCORE_MAT: f32 = 1000.0;

/// Au-delà de ce seuil (en valeur absolue) un score est un score de MAT :
/// les valeurs réseau vivent dans [-1, 1] et les mats dans
/// ±[SCORE_MAT - MAX_PLY - PROF_QUIESCENCE, SCORE_MAT], les deux échelles ne se
/// croisent jamais (1000 - 128 = 872 > 800).
const SEUIL_MAT: f32 = 800.0;

/// Profondeur maximale de l'approfondissement itératif (max_profondeur = 0 → ∞
/// borné par cette valeur, les limites de nœuds/temps arrêtant bien avant).
const PROF_MAX: u32 = 64;

/// Ply maximal de la recherche principale (garde-fou : au-delà on bascule en
/// quiescence). Doit rester > PROF_MAX et < 256 pour l'échelle des mats.
const MAX_PLY: usize = 120;

/// Profondeur maximale de la quiescence (prises en cascade bornées).
const PROF_QUIESCENCE: u32 = 8;

/// Réduction du null-move (profondeur fille = profondeur - 1 - R_NULL).
const R_NULL: u32 = 2;

/// Largeur de la fenêtre nulle du null-move : les scores sont des f32 continus,
/// on teste « >= beta » avec une fenêtre [beta - EPS_NUL, beta].
const EPS_NUL: f32 = 1e-3;

/// Le chrono (Instant::now) n'est consulté que tous les ~1024 nœuds : un appel
/// d'horloge par nœud coûterait plus cher que le nœud lui-même.
const INTERVALLE_CHRONO: u64 = 1024;

// --- Table de transposition -------------------------------------------------

const DRAPEAU_VIDE: u8 = 0;
/// Score exact (la fenêtre n'a pas été coupée).
const DRAPEAU_EXACT: u8 = 1;
/// Borne inférieure : coupure bêta, le score réel est >= au score stocké.
const DRAPEAU_BORNE_INF: u8 = 2;
/// Borne supérieure : aucun coup n'a dépassé alpha, le score réel est <= stocké.
const DRAPEAU_BORNE_SUP: u8 = 3;

/// Coup compacté « aucun » : from=a1, to=a1, promotion=0 n'est jamais un coup
/// légal, 0 sert donc de sentinelle.
const COUP_AUCUN: u16 = 0;

/// Entrée de la table de transposition (16 octets, sans padding).
///
/// ATTENTION AUX MATS (le piège classique) : un score de mat vaut
/// ±(SCORE_MAT - ply_racine), il dépend donc de la distance à la RACINE de la
/// recherche en cours. Stocké tel quel, il serait faux relu depuis un nœud à un
/// autre ply (le même mat semblerait plus proche ou plus lointain qu'il ne
/// l'est). On stocke donc les mats convertis en « distance au NŒUD » :
///   au stockage  : score_tt = score + ply (mats gagnants), score - ply (perdants)
///   à la relecture : score  = score_tt - ply', score_tt + ply'
/// si bien qu'une entrée écrite à ply 3 et relue à ply 7 rend un score de mat
/// correct vu de la nouvelle racine. Voir score_vers_tt / score_depuis_tt.
#[derive(Clone, Copy)]
struct EntreeTT {
    cle: u64,
    score: f32,
    coup: u16,
    profondeur: u8,
    drapeau: u8,
}

const ENTREE_VIDE: EntreeTT = EntreeTT {
    cle: 0,
    score: 0.0,
    coup: COUP_AUCUN,
    profondeur: 0,
    drapeau: DRAPEAU_VIDE,
};

/// Conversion d'un score « vu de la racine » en score « vu du nœud » pour le
/// stockage en TT (voir le commentaire d'EntreeTT : c'est LE piège des mats).
fn score_vers_tt(score: f32, ply: u32) -> f32 {
    if score > SEUIL_MAT {
        score + ply as f32
    } else if score < -SEUIL_MAT {
        score - ply as f32
    } else {
        score
    }
}

/// Conversion inverse à la relecture : le mat stocké « à distance du nœud »
/// redevient un score vu de la racine de la recherche EN COURS.
fn score_depuis_tt(score: f32, ply: u32) -> f32 {
    if score > SEUIL_MAT {
        score - ply as f32
    } else if score < -SEUIL_MAT {
        score + ply as f32
    } else {
        score
    }
}

// --- Aides génériques -------------------------------------------------------

/// Hachage zobrist 64 bits (même convention que selfplay/arena : mode Legal).
fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Compacte un coup en 16 bits : from (6) | to (6) | promotion (3).
/// Suffit à identifier un coup PARMI LES COUPS LÉGAUX d'une position (roque :
/// from = case du roi, to = case de la tour — paire unique elle aussi). Le coup
/// TT n'est jamais joué directement, seulement comparé à la liste légale.
fn compacter(m: &Move) -> u16 {
    let from = m.from().map_or(0, usize::from) as u16;
    let to = usize::from(m.to()) as u16;
    let promo = m.promotion().map_or(0u16, u16::from);
    from | (to << 6) | (promo << 12)
}

/// Valeur d'ordre d'un rôle : P=1, N=2, B=3, R=4, Q=5, K=6 (l'ordre suffit,
/// on ne s'en sert que pour trier).
fn valeur_role(r: shakmaty::Role) -> i32 {
    usize::from(r) as i32
}

/// Clé MVV-LVA d'un coup tactique : victime la plus grosse d'abord (facteur
/// 16 pour que la victime domine), agresseur le plus léger ensuite — c'est le
/// « valeur victime - valeur agresseur » du cahier des charges, à l'échelle
/// près qui évite qu'un gros agresseur déclasse une grosse victime.
/// Les promotions comptent comme tactiques (dame d'abord).
fn cle_mvv_lva(m: &Move) -> i32 {
    let mut cle = 0;
    if let Some(victime) = m.capture() {
        cle += 100_000 + 16 * valeur_role(victime) - valeur_role(m.role());
    }
    if let Some(promo) = m.promotion() {
        cle += 90_000 + 16 * valeur_role(promo);
    }
    cle
}

/// Le trait possède-t-il au moins une pièce qui ne soit ni pion ni roi ?
/// (Condition du null-move : évite les positions de zugzwang de finale de
/// pions, où « passer » est souvent la meilleure option réelle.)
fn a_piece_non_pion(pos: &Chess) -> bool {
    (pos.us() & !(pos.board().pawns() | pos.board().kings())).any()
}

fn idx_couleur(c: Color) -> usize {
    match c {
        Color::White => 0,
        Color::Black => 1,
    }
}

/// Limites d'un appel de recherche. 0 = pas de limite pour ce critère
/// (au moins un critère doit être non nul).
#[derive(Clone, Copy)]
pub struct Limites {
    pub max_noeuds: u64,
    pub max_profondeur: u32,
    pub movetime_ms: u64,
}

pub struct Resultat {
    /// Meilleur coup (None seulement sans coup légal).
    pub coup: Option<Move>,
    /// Score du point de vue du trait ([-1,1] hors mats, ±(SCORE_MAT-ply) sinon).
    pub score: f32,
    /// Profondeur complète atteinte par l'approfondissement itératif.
    pub profondeur: u32,
    pub noeuds: u64,
    /// Scores racine (coup, score) de la DERNIÈRE itération complète —
    /// sert à l'échantillonnage en température du self-play.
    pub scores_racine: Vec<(Move, f32)>,
}

/// État persistant d'un chercheur : table de transposition, killers,
/// historique. UN chercheur par thread (rien de partagé). Possède son réseau
/// via Arc (partagé en lecture entre threads).
pub struct Recherche {
    pub net: Arc<Mlp>,
    /// Table de transposition : 2^taille_tt_log2 entrées, index = cle & masque.
    tt: Vec<EntreeTT>,
    masque: u64,
    /// Deux killers par ply (coups calmes ayant produit une coupure bêta).
    killers: Vec<[u16; 2]>,
    /// Historique [couleur][from][to] aplati (2 × 64 × 64), incrémenté de
    /// profondeur² à chaque coupure bêta d'un coup calme.
    historique: Vec<u32>,
    /// Tampon d'encodage réutilisé pour chaque évaluation réseau.
    tampon: Vec<f32>,
    /// Clés zobrist de la LIGNE en cours d'exploration (racine → nœud) :
    /// détection des répétitions dans l'arbre. Pile poussée/poppée par negamax.
    chemin: Vec<u64>,
    // --- État d'un appel de cherche() (réinitialisé à chaque appel) ---
    noeuds: u64,
    stop: bool,
    /// Faux pendant l'itération 1 : elle est TOUJOURS menée à terme pour
    /// garantir un coup, quelles que soient les limites.
    limites_actives: bool,
    limite_noeuds: u64,
    fin: Option<Instant>,
    prochaine_verif_chrono: u64,
}

impl Recherche {
    /// `taille_tt_log2` : nombre d'entrées de la table = 2^n (ex. 20 → ~1M
    /// d'entrées). La table est allouée une fois et réutilisée entre coups
    /// (les hits entre coups successifs sont une grosse part du gain).
    pub fn new(net: Arc<Mlp>, taille_tt_log2: u32) -> Self {
        assert!(taille_tt_log2 <= 30, "taille_tt_log2 déraisonnable (> 2^30 entrées)");
        let n = 1usize << taille_tt_log2;
        Recherche {
            net,
            tt: vec![ENTREE_VIDE; n],
            masque: (n - 1) as u64,
            killers: vec![[COUP_AUCUN; 2]; MAX_PLY],
            historique: vec![0; 2 * 64 * 64],
            tampon: vec![0.0; N_FEATURES],
            chemin: Vec::with_capacity(MAX_PLY + 1),
            noeuds: 0,
            stop: false,
            limites_actives: false,
            limite_noeuds: u64::MAX,
            fin: None,
            prochaine_verif_chrono: 0,
        }
    }

    /// Approfondissement itératif 1..=max jusqu'à épuisement des limites.
    /// Doit gérer : quiescence aux feuilles (prises + promotions, stand-pat
    /// réseau), null-move (R=2, pas en échec, matériel non-pion présent),
    /// tri coup TT > prises MVV-LVA > killers > historique, mats/pats exacts,
    /// nulles (50 coups, matériel insuffisant) à 0. La détection de répétition
    /// DANS l'arbre n'est pas exigée (les boucles de jeu l'arbitrent).
    pub fn cherche(&mut self, pos: &Chess, limites: Limites) -> Resultat {
        assert!(
            limites.max_noeuds > 0 || limites.max_profondeur > 0 || limites.movetime_ms > 0,
            "Limites : au moins un critère doit être non nul"
        );
        self.noeuds = 0;
        self.stop = false;
        self.limites_actives = false;
        self.limite_noeuds = if limites.max_noeuds == 0 { u64::MAX } else { limites.max_noeuds };
        self.fin = (limites.movetime_ms > 0)
            .then(|| Instant::now() + Duration::from_millis(limites.movetime_ms));
        self.prochaine_verif_chrono = INTERVALLE_CHRONO;

        let coups = pos.legal_moves();
        if coups.is_empty() {
            // Position terminale : mat (le trait perd, ply 0) ou pat.
            return Resultat {
                coup: None,
                score: if pos.is_check() { -SCORE_MAT } else { 0.0 },
                profondeur: 0,
                noeuds: 0,
                scores_racine: Vec::new(),
            };
        }
        // NB : si la position est déjà nulle « aux règles » (50 coups, matériel
        // insuffisant), l'arbitrage appartient aux boucles de jeu ; ici on rend
        // quand même un coup (contrat : None seulement sans coup légal).

        let prof_max = if limites.max_profondeur == 0 {
            PROF_MAX
        } else {
            limites.max_profondeur.min(PROF_MAX)
        };

        // Ordre racine initial : coup TT de la recherche précédente (grosse
        // source de gain entre coups successifs d'une même partie), puis
        // tactiques MVV-LVA, puis historique.
        let cle_racine = zobrist(pos);
        // La racine ouvre la ligne courante (les nœuds la prolongent) ; un
        // arrêt en plein arbre peut laisser des résidus → on repart propre.
        self.chemin.clear();
        self.chemin.push(cle_racine);
        let coup_tt_racine = self.sonde(cle_racine).map_or(COUP_AUCUN, |e| e.coup);
        let couleur = pos.turn();
        let mut ordre: Vec<Move> = coups.iter().cloned().collect();
        ordre.sort_by_cached_key(|m| Reverse(self.cle_ordre(m, coup_tt_racine, 0, couleur)));

        // Dernière itération COMPLÈTE : (meilleur coup, score, profondeur, scores racine).
        let mut complete: Option<(Move, f32, u32, Vec<(Move, f32)>)> = None;

        for d in 1..=prof_max {
            // L'itération 1 ignore les limites : il faut TOUJOURS un coup.
            self.limites_actives = d > 1;
            if d > 1 && self.budget_epuise() {
                break;
            }

            let mut alpha = f32::NEG_INFINITY;
            let mut best = f32::NEG_INFINITY;
            let mut meilleur: Option<Move> = None;
            let mut scores_iter: Vec<(Move, f32)> = Vec::with_capacity(ordre.len());
            let mut interrompue = false;

            for m in &ordre {
                let mut fille = pos.clone();
                fille.play_unchecked(m);
                // Fenêtre racine : alpha monte, bêta reste infini (tous les
                // coups racine sont cherchés). Le meilleur score est exact ;
                // les autres peuvent n'être que des bornes supérieures
                // (fail-soft) — suffisant pour l'échantillonnage en
                // température, qui ne sert qu'à l'exploration.
                let v = -self.negamax(&fille, d - 1, 1, f32::NEG_INFINITY, -alpha, false);
                if self.stop {
                    interrompue = true;
                    break;
                }
                // Chrono aussi consulté entre deux coups racine : la granularité
                // en nœuds (INTERVALLE_CHRONO) peut être trop grossière quand le
                // réseau est lent ; ici l'appel d'horloge est gratuit à l'échelle
                // d'un sous-arbre. Sans incidence sur le déterminisme à budget de
                // nœuds fixe (fin = None dans ce cas).
                if self.limites_actives && self.fin.is_some_and(|f| Instant::now() >= f) {
                    self.stop = true;
                    interrompue = true;
                    break;
                }
                scores_iter.push((m.clone(), v));
                if v > best {
                    best = v;
                    meilleur = Some(m.clone());
                }
                if v > alpha {
                    alpha = v;
                }
            }
            if interrompue {
                break; // itération jetée : on garde la dernière complète
            }

            // Réordonne la racine pour l'itération suivante : meilleurs scores
            // d'abord (tri stable : le meilleur reste devant ses ex æquo).
            let mut tri = scores_iter.clone();
            tri.sort_by(|a, b| b.1.total_cmp(&a.1));
            ordre = tri.into_iter().map(|(m, _)| m).collect();

            let coup = meilleur.expect("itération complète avec coups légaux");
            self.stocke(cle_racine, d, DRAPEAU_EXACT, score_vers_tt(best, 0), compacter(&coup));
            let mat_trouve = best.abs() > SEUIL_MAT;
            complete = Some((coup, best, d, scores_iter));
            if mat_trouve {
                break; // mat prouvé dans l'horizon : inutile de creuser
            }
        }

        let (coup, score, profondeur, scores_racine) =
            complete.expect("l'itération 1 est toujours menée à terme");
        Resultat {
            coup: Some(coup),
            score,
            profondeur,
            noeuds: self.noeuds,
            scores_racine,
        }
    }

    /// À appeler entre deux PARTIES (pas entre deux coups) : vide TT,
    /// killers et historique.
    pub fn nouvelle_partie(&mut self) {
        self.tt.fill(ENTREE_VIDE);
        self.killers.fill([COUP_AUCUN; 2]);
        self.historique.fill(0);
    }

    // --- Négamax alpha-bêta (fail-soft) --------------------------------------

    fn negamax(
        &mut self,
        pos: &Chess,
        profondeur: u32,
        ply: u32,
        mut alpha: f32,
        beta: f32,
        null_interdit: bool,
    ) -> f32 {
        // Feuille : quiescence (elle fait son propre comptage et ses propres
        // verdicts mat/pat/nulle, avec sa propre génération de coups).
        if profondeur == 0 || ply as usize >= MAX_PLY {
            return self.quiesce(pos, ply, PROF_QUIESCENCE, alpha, beta);
        }

        self.noeuds += 1;
        if self.verifier_arret() {
            return 0.0; // valeur jetée : l'itération interrompue est abandonnée
        }

        // Génération COMPLÈTE : tous les coups sont joués ici, le verdict de
        // mat/pat est donc exact. Mat/pat testés AVANT la règle des 50 coups,
        // comme partout dans le projet : un mat délivré pile au 100e
        // demi-coup reste un mat.
        let coups = pos.legal_moves();
        let en_echec = pos.is_check();
        if coups.is_empty() {
            return if en_echec { -(SCORE_MAT - ply as f32) } else { 0.0 };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
            return 0.0;
        }

        let cle = zobrist(pos);

        // Répétition DANS LA LIGNE explorée (la position est déjà apparue
        // entre la racine et ce nœud) → nulle, 0.0. Le contrat n'exige pas la
        // détection de la 3e répétition RÉELLE (les boucles de jeu
        // l'arbitrent), mais sans CE test un camp gagnant « mélange » ses
        // pièces vers la répétition au lieu de progresser — la 2e occurrence
        // dans une même ligne vaut nulle, comme dans tous les moteurs. Ce
        // score dépend du chemin : il est rendu AVANT la sonde TT et n'est
        // jamais stocké (le reste de contamination indirecte via les
        // sous-arbres est le compromis GHI standard).
        if self.chemin.contains(&cle) {
            return 0.0;
        }

        // Sonde de la table de transposition. NB : la clé zobrist ignore le
        // compteur des 50 coups — une entrée peut donc court-circuiter un
        // sous-arbre qui aurait buté sur la règle ; c'est le compromis
        // standard de tous les moteurs, accepté ici aussi.
        let mut coup_tt = COUP_AUCUN;
        if let Some(e) = self.sonde(cle) {
            coup_tt = e.coup;
            if u32::from(e.profondeur) >= profondeur {
                // Relecture avec ré-ajustement du ply : voir EntreeTT (mats).
                let s = score_depuis_tt(e.score, ply);
                match e.drapeau {
                    DRAPEAU_EXACT => return s,
                    DRAPEAU_BORNE_INF if s >= beta => return s,
                    DRAPEAU_BORNE_SUP if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        let alpha_orig = alpha;
        // La clé du nœud rejoint la ligne courante le temps d'explorer ses
        // enfants (null-move compris) ; UNIQUE pop après le bloc, quel que
        // soit le chemin de sortie.
        self.chemin.push(cle);
        let (best, meilleur_coup, stocker) = 'corps: {
            // Null-move (R=2) : si « passer » suffit déjà à couper, la
            // position est si bonne qu'on s'épargne la recherche complète.
            // Conditions : pas en échec, profondeur >= 3, du matériel non-pion
            // (zugzwang), jamais deux nulls consécutifs, une bêta finie
            // (sinon rien à couper) et une bêta HORS zone de mat (garde
            // standard des moteurs : si beta > SEUIL_MAT — fenêtre d'un
            // sous-arbre de preuve de mat — le rabattement anti-mat du
            // fail-high renverrait beta, c'est-à-dire un score de MAT non
            // prouvé, et une défense réelle pourrait être élaguée en cas de
            // zugzwang).
            if !null_interdit
                && profondeur >= 1 + R_NULL
                && !en_echec
                && beta.is_finite()
                && beta < SEUIL_MAT
                && a_piece_non_pion(pos)
            {
                if let Ok(passe) = pos.clone().swap_turn() {
                    let v = -self.negamax(
                        &passe,
                        profondeur - 1 - R_NULL,
                        ply + 1,
                        -beta,
                        -beta + EPS_NUL,
                        true,
                    );
                    if self.stop {
                        break 'corps (0.0, COUP_AUCUN, false);
                    }
                    if v >= beta {
                        // Jamais de score de MAT non prouvé issu d'un
                        // null-move : on rabat sur beta (mater un adversaire
                        // qui passe son tour ne prouve pas un mat réel) —
                        // et beta < SEUIL_MAT (garde d'entrée), donc la
                        // valeur rendue est toujours hors zone de mat.
                        break 'corps (if v > SEUIL_MAT { beta } else { v }, COUP_AUCUN, false);
                    }
                }
            }

            // Tri : coup TT > tactiques MVV-LVA > killers du ply > historique.
            let mut ordonnes: Vec<(i32, &Move)> = coups
                .iter()
                .map(|m| (self.cle_ordre(m, coup_tt, ply, pos.turn()), m))
                .collect();
            ordonnes.sort_unstable_by_key(|(k, _)| Reverse(*k));

            let mut best = f32::NEG_INFINITY;
            let mut meilleur_coup = COUP_AUCUN;

            for (_, m) in &ordonnes {
                let mut fille = pos.clone();
                fille.play_unchecked(m);
                let v = -self.negamax(&fille, profondeur - 1, ply + 1, -beta, -alpha, false);
                if self.stop {
                    // Valeur jetée, surtout ne rien stocker en TT.
                    break 'corps (best, COUP_AUCUN, false);
                }
                if v > best {
                    best = v;
                    meilleur_coup = compacter(m);
                    if v > alpha {
                        alpha = v;
                        if alpha >= beta {
                            // Coupure bêta : killers + historique (coups calmes).
                            if !m.is_capture() && !m.is_promotion() {
                                self.note_killer(ply, meilleur_coup);
                                self.note_historique(pos.turn(), m, profondeur);
                            }
                            break;
                        }
                    }
                }
            }
            (best, meilleur_coup, true)
        };
        self.chemin.pop();
        if !stocker {
            return best;
        }

        // Stockage TT : drapeau selon la fenêtre d'ORIGINE, score de mat
        // converti en distance au nœud (score_vers_tt, voir EntreeTT).
        let drapeau = if best >= beta {
            DRAPEAU_BORNE_INF
        } else if best <= alpha_orig {
            DRAPEAU_BORNE_SUP
        } else {
            DRAPEAU_EXACT
        };
        self.stocke(cle, profondeur, drapeau, score_vers_tt(best, ply), meilleur_coup);
        best
    }

    // --- Quiescence ----------------------------------------------------------

    /// Hors échec : stand-pat réseau puis prises et promotions uniquement,
    /// triées MVV-LVA, alpha-bêta fail-soft. EN échec : pas de stand-pat (on
    /// ne peut pas « passer » sous échec), on cherche TOUTES les évasions — la
    /// liste légale complète EST la liste des évasions, donc le mat est déclaré
    /// EXACTEMENT même ici (le piège « conclure au mat à court de prises »
    /// ne peut pas se produire : à court de prises hors échec on rend le
    /// stand-pat, et en échec tous les coups sont essayés).
    fn quiesce(&mut self, pos: &Chess, ply: u32, prof_restante: u32, mut alpha: f32, beta: f32) -> f32 {
        self.noeuds += 1;
        if self.verifier_arret() {
            return 0.0;
        }

        // Liste légale COMPLÈTE (c'est ainsi que shakmaty produit les prises) :
        // vide → mat ou pat, verdict exact même en quiescence.
        let coups = pos.legal_moves();
        let en_echec = pos.is_check();
        if coups.is_empty() {
            return if en_echec { -(SCORE_MAT - ply as f32) } else { 0.0 };
        }
        if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
            return 0.0;
        }
        let mut best;
        if en_echec {
            // Garde-fou de profondeur : une cascade d'échecs ne peut pas
            // s'étendre indéfiniment, on retombe sur l'évaluation brute.
            if prof_restante == 0 {
                return self.evaluer(pos);
            }
            best = f32::NEG_INFINITY; // pas de stand-pat sous échec
        } else {
            // Stand-pat : « je peux m'abstenir de prendre » — l'évaluation
            // réseau de la position telle quelle, perspective du trait.
            let stand_pat = self.evaluer(pos);
            if prof_restante == 0 {
                return stand_pat; // profondeur de quiescence épuisée
            }
            if stand_pat >= beta {
                return stand_pat; // fail-soft
            }
            if stand_pat > alpha {
                alpha = stand_pat;
            }
            best = stand_pat;
        }

        // En échec : toutes les évasions ; sinon tactiques seulement.
        // Tri MVV-LVA, plus grosse victime d'abord (les évasions calmes,
        // clé 0, passent après les prises).
        let mut a_jouer: Vec<(i32, &Move)> = coups
            .iter()
            .filter(|m| en_echec || m.is_capture() || m.is_promotion())
            .map(|m| (cle_mvv_lva(m), m))
            .collect();
        a_jouer.sort_unstable_by_key(|(k, _)| Reverse(*k));

        for (_, m) in &a_jouer {
            let mut fille = pos.clone();
            fille.play_unchecked(m);
            let v = -self.quiesce(&fille, ply + 1, prof_restante - 1, -beta, -alpha);
            if self.stop {
                return best; // valeur jetée de toute façon (itération abandonnée)
            }
            if v > best {
                best = v;
                if v > alpha {
                    alpha = v;
                    if alpha >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    // --- Évaluation réseau ---------------------------------------------------

    /// Encode la position dans le tampon réutilisé puis passe avant du réseau.
    /// Sortie dans [-1, 1], perspective du trait (convention du projet).
    fn evaluer(&mut self, pos: &Chess) -> f32 {
        encode(pos, &mut self.tampon);
        self.net.forward_one(&self.tampon)
    }

    // --- Limites -------------------------------------------------------------

    /// Vrai si la recherche doit s'arrêter. Le budget de nœuds est testé à
    /// CHAQUE nœud (comparaison d'entiers, gratuite et déterministe) ; le
    /// chrono seulement tous les ~INTERVALLE_CHRONO nœuds.
    fn verifier_arret(&mut self) -> bool {
        if self.stop {
            return true;
        }
        if !self.limites_actives {
            return false; // itération 1 : toujours menée à terme
        }
        if self.noeuds >= self.limite_noeuds {
            self.stop = true;
            return true;
        }
        if let Some(fin) = self.fin {
            if self.noeuds >= self.prochaine_verif_chrono {
                self.prochaine_verif_chrono = self.noeuds + INTERVALLE_CHRONO;
                if Instant::now() >= fin {
                    self.stop = true;
                    return true;
                }
            }
        }
        false
    }

    /// Test direct (entre deux itérations) : budget déjà consommé ?
    fn budget_epuise(&self) -> bool {
        self.noeuds >= self.limite_noeuds
            || self.fin.is_some_and(|f| Instant::now() >= f)
    }

    // --- Table de transposition ----------------------------------------------

    fn sonde(&self, cle: u64) -> Option<EntreeTT> {
        let e = self.tt[(cle & self.masque) as usize];
        (e.drapeau != DRAPEAU_VIDE && e.cle == cle).then_some(e)
    }

    /// Remplacement : case vide, clé différente, ou profondeur >= existante
    /// (une recherche plus profonde de la même position est plus fiable).
    fn stocke(&mut self, cle: u64, profondeur: u32, drapeau: u8, score: f32, coup: u16) {
        let e = &mut self.tt[(cle & self.masque) as usize];
        if e.drapeau == DRAPEAU_VIDE || e.cle != cle || profondeur >= u32::from(e.profondeur) {
            *e = EntreeTT {
                cle,
                score,
                coup,
                profondeur: profondeur.min(255) as u8,
                drapeau,
            };
        }
    }

    // --- Tri des coups -------------------------------------------------------

    /// Clé de tri décroissante d'un coup dans la recherche principale :
    /// coup TT (1 000 000) > tactiques MVV-LVA (~100 000-190 000)
    /// > promotions calmes (~90 000) > killers (80 000 / 79 000)
    /// > historique (0..=60 000).
    fn cle_ordre(&self, m: &Move, coup_tt: u16, ply: u32, couleur: Color) -> i32 {
        let c = compacter(m);
        if coup_tt != COUP_AUCUN && c == coup_tt {
            return 1_000_000;
        }
        let cle = cle_mvv_lva(m);
        if cle != 0 {
            return cle; // prise et/ou promotion
        }
        let k = &self.killers[ply as usize];
        if c == k[0] {
            return 80_000;
        }
        if c == k[1] {
            return 79_000;
        }
        self.historique[Self::idx_historique(couleur, m)].min(60_000) as i32
    }

    fn idx_historique(couleur: Color, m: &Move) -> usize {
        let from = m.from().map_or(0, usize::from);
        let to = usize::from(m.to());
        idx_couleur(couleur) * 64 * 64 + from * 64 + to
    }

    fn note_killer(&mut self, ply: u32, coup: u16) {
        let k = &mut self.killers[ply as usize];
        if k[0] != coup {
            k[1] = k[0];
            k[0] = coup;
        }
    }

    /// Historique : +profondeur² à la coupure bêta. En cas de débordement
    /// (rarissime), toute la table est divisée par 2 — les ordres relatifs
    /// sont conservés.
    fn note_historique(&mut self, couleur: Color, m: &Move, profondeur: u32) {
        let idx = Self::idx_historique(couleur, m);
        self.historique[idx] += profondeur * profondeur;
        if self.historique[idx] > 1_000_000 {
            for h in &mut self.historique {
                *h /= 2;
            }
        }
    }
}

// --- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{Bot, NetBot};
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use shakmaty::fen::Fen;
    use shakmaty::{CastlingMode, Square};
    use std::collections::HashMap;

    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    /// Réseau réduit DÉTERMINISTE dérivé de Mlp::new(0) : mêmes 773 entrées,
    /// mais 12 neurones cachés (troncature des poids du réseau neuf).
    ///
    /// POURQUOI : en profil dev (cargo test --lib, sans optimisations), un
    /// forward du réseau complet 773→512→64→1 coûte ~7 ms ; une recherche de
    /// 3 000 nœuds en ferait ~2 000 → l'arène du test (4) durerait des heures.
    /// Le réseau réduit (~13 000 multiplications) garde tout ce qui compte
    /// pour ces tests : une évaluation non triviale, déterministe, identique
    /// pour les deux camps. Le réseau complet reste couvert par le test de
    /// fumée `reseau_complet_fumee`.
    fn reseau_reduit() -> Arc<Mlp> {
        let base = Mlp::new(0);
        let (n_in, cache) = (N_FEATURES, 12usize);
        let w0: Vec<f32> = base.weights[0][..cache * n_in].to_vec();
        let b0: Vec<f32> = base.biases[0][..cache].to_vec();
        let w1: Vec<f32> = base.weights[1][..cache].to_vec();
        let b1: Vec<f32> = vec![base.biases[2][0]];
        let formes_w = [w0.len(), w1.len()];
        let formes_b = [b0.len(), b1.len()];
        Arc::new(Mlp {
            sizes: vec![n_in, cache, 1],
            weights: vec![w0, w1],
            biases: vec![b0, b1],
            adam_mw: formes_w.iter().map(|&n| vec![0.0; n]).collect(),
            adam_vw: formes_w.iter().map(|&n| vec![0.0; n]).collect(),
            adam_mb: formes_b.iter().map(|&n| vec![0.0; n]).collect(),
            adam_vb: formes_b.iter().map(|&n| vec![0.0; n]).collect(),
            steps: 0,
        })
    }

    /// Réseau linéaire [773 → 1] aux poids MATÉRIELS (nos P,N,B,R,Q positifs,
    /// les leurs négatifs, roques/en-passant à 0) : une évaluation jouable,
    /// déterministe et quasi gratuite en profil dev. L'arène du test (4)
    /// l'utilise pour les DEUX camps : le duel mesure alors l'apport de la
    /// RECHERCHE (tactique profonde, mats exacts) à savoir égal — un réseau
    /// purement aléatoire ne donne qu'un effet Beal marginal (~70 % mesuré),
    /// trop juste pour servir de garde-fou fiable ; le réseau NEUF reste
    /// couvert par les autres tests (mats, déterminisme, TT, fumée).
    fn reseau_materiel() -> Arc<Mlp> {
        let valeurs = [0.10f32, 0.30, 0.32, 0.50, 0.90, 0.0];
        let mut w0 = vec![0.0f32; N_FEATURES];
        for (plan, &v) in valeurs.iter().enumerate() {
            for case in 0..64 {
                w0[plan * 64 + case] = v; // nos pièces
                w0[(6 + plan) * 64 + case] = -v; // les leurs
            }
        }
        Arc::new(Mlp {
            sizes: vec![N_FEATURES, 1],
            weights: vec![w0],
            biases: vec![vec![0.0]],
            adam_mw: vec![vec![0.0; N_FEATURES]],
            adam_vw: vec![vec![0.0; N_FEATURES]],
            adam_mb: vec![vec![0.0]],
            adam_vb: vec![vec![0.0]],
            steps: 0,
        })
    }

    fn limites_prof(p: u32) -> Limites {
        Limites { max_noeuds: 0, max_profondeur: p, movetime_ms: 0 }
    }

    fn limites_noeuds(n: u64) -> Limites {
        Limites { max_noeuds: n, max_profondeur: 0, movetime_ms: 0 }
    }

    /// (1) Mat en 1 : tour a1, roi noir g8 enfermé par ses pions → Ra8#.
    #[test]
    fn mat_en_1_trouve() {
        let pos = pos_de_fen("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1");
        let mut r = Recherche::new(reseau_reduit(), 14);
        let res = r.cherche(&pos, limites_prof(2));
        let coup = res.coup.expect("un coup légal existe");
        assert!(res.score > 900.0, "score de mat attendu, obtenu {}", res.score);
        assert_eq!(coup.to(), Square::A8, "Ra8# attendu, obtenu {coup:?}");
        // Mat en 1 exactement : SCORE_MAT - 1.
        assert!((res.score - (SCORE_MAT - 1.0)).abs() < 1e-3);
    }

    /// (2) Mat en 2 (mat du couloir à deux tours) trouvé à profondeur 4 :
    /// tours a2 et b1 contre roi h8 (ex. 1.Ra7 Kg8 2.Rb8#, plusieurs échelles
    /// gagnent). Aucun mat en 1 n'existe : le score doit valoir exactement
    /// SCORE_MAT - 3 (mat au 3e demi-coup), et le coup choisi doit réellement
    /// forcer le mat : la position fille, cherchée à son tour, vaut
    /// -(SCORE_MAT - 2) pour le camp maté.
    #[test]
    fn mat_en_2_trouve_profondeur_4() {
        let pos = pos_de_fen("7k/8/8/8/8/8/R7/1R5K w - - 0 1");
        let mut r = Recherche::new(reseau_reduit(), 16);
        let res = r.cherche(&pos, limites_prof(4));
        let coup = res.coup.expect("un coup légal existe");
        assert!(
            (res.score - (SCORE_MAT - 3.0)).abs() < 1e-3,
            "mat en 2 (score {}) attendu, obtenu {}",
            SCORE_MAT - 3.0,
            res.score
        );
        assert!(res.profondeur >= 3, "profondeur {} < 3", res.profondeur);
        // Contre-vérification : après le coup choisi, le camp adverse est bien
        // maté en 2 demi-coups quoi qu'il joue.
        let fille = pos.play(&coup).expect("coup légal");
        let mut rv = Recherche::new(reseau_reduit(), 14);
        let verdict = rv.cherche(&fille, limites_prof(3));
        assert!(
            (verdict.score + (SCORE_MAT - 2.0)).abs() < 1e-3,
            "le coup {coup:?} ne force pas le mat : verdict {}",
            verdict.score
        );
    }

    /// (3) Déterminisme : mêmes limites de nœuds, deux chercheurs neufs →
    /// résultats identiques bit à bit (aucune horloge en jeu).
    #[test]
    fn deterministe_a_noeuds_fixes() {
        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let net = reseau_reduit();
        let mut r1 = Recherche::new(net.clone(), 14);
        let mut r2 = Recherche::new(net, 14);
        let a = r1.cherche(&pos, limites_noeuds(1500));
        let b = r2.cherche(&pos, limites_noeuds(1500));
        assert_eq!(a.coup, b.coup);
        assert_eq!(a.score, b.score);
        assert_eq!(a.profondeur, b.profondeur);
        assert_eq!(a.noeuds, b.noeuds);
        assert_eq!(a.scores_racine.len(), b.scores_racine.len());
        for ((ma, va), (mb, vb)) in a.scores_racine.iter().zip(&b.scores_racine) {
            assert_eq!(ma, mb);
            assert_eq!(va, vb);
        }
    }

    /// Joue une partie chercheur (3000 nœuds) contre NetBot 1 pli, même réseau.
    /// `ouverture` : plis initiaux joués au hasard (les deux bots étant quasi
    /// déterministes, c'est elle qui diversifie les parties). Renvoie le
    /// résultat côté chercheur : 1 victoire, 0.5 nulle, 0 défaite.
    fn partie_recherche_contre_1_pli(
        recherche: &mut Recherche,
        net: &Mlp,
        chercheur_blanc: bool,
        graine: u64,
        ouverture: &[Move],
    ) -> (f32, u32) {
        let mut pos = Chess::default();
        let mut repetitions: HashMap<u64, u8> = HashMap::new();
        repetitions.insert(zobrist(&pos), 1);
        for m in ouverture {
            pos = pos.play(m).expect("coup d'ouverture légal");
            *repetitions.entry(zobrist(&pos)).or_insert(0) += 1;
        }
        recherche.nouvelle_partie();
        let mut adversaire = NetBot::new(net, graine, 0.0, 1);
        let limites = limites_noeuds(3000);
        let mut plies = 0u32;

        let resultat_blancs = loop {
            let coups = pos.legal_moves();
            if coups.is_empty() {
                break if pos.is_check() {
                    if pos.turn() == Color::White { -1.0 } else { 1.0 }
                } else {
                    0.0
                };
            }
            if pos.is_insufficient_material() || pos.halfmoves() >= 100 || plies >= 200 {
                break 0.0;
            }
            let tour_chercheur = (pos.turn() == Color::White) == chercheur_blanc;
            let m = if tour_chercheur {
                recherche.cherche(&pos, limites).coup.expect("coup légal")
            } else {
                adversaire.choose(&pos).expect("coup légal")
            };
            pos = pos.play(&m).expect("coup légal");
            plies += 1;
            let c = repetitions.entry(zobrist(&pos)).or_insert(0);
            *c += 1;
            if *c >= 3 {
                break 0.0;
            }
        };
        let cote = if chercheur_blanc { resultat_blancs } else { -resultat_blancs };
        ((cote + 1.0) / 2.0, plies)
    }

    /// (4) La recherche à 3000 nœuds bat le réseau brut (1 pli, MÊME réseau
    /// pour les deux camps) : mini-arène de 10 parties, ouvertures aléatoires
    /// appariées (couleurs échangées), score exigé >= 70 %. Voir
    /// reseau_materiel() pour le choix du réseau de duel.
    #[test]
    fn recherche_bat_reseau_brut_mini_arene() {
        let net = reseau_materiel();
        let mut recherche = Recherche::new(net.clone(), 16);
        let mut points = 0.0f32;
        for paire in 0..5u64 {
            // Ouverture aléatoire de 4 plis, partagée par les deux parties de
            // la paire (équité : chaque camp joue la même ouverture des deux
            // couleurs).
            let mut rng = StdRng::seed_from_u64(0xA5E5 + paire);
            let mut pos = Chess::default();
            let mut ouverture = Vec::new();
            for _ in 0..4 {
                let m = pos
                    .legal_moves()
                    .choose(&mut rng)
                    .cloned()
                    .expect("ouverture jouable");
                pos = pos.play(&m).expect("coup légal");
                ouverture.push(m);
            }
            for (i, chercheur_blanc) in [(0u64, true), (1u64, false)] {
                let (pts, plies) = partie_recherche_contre_1_pli(
                    &mut recherche,
                    &net,
                    chercheur_blanc,
                    7000 + paire * 2 + i,
                    &ouverture,
                );
                println!(
                    "  partie {} (chercheur {}) : {} en {} plis",
                    paire * 2 + i,
                    if chercheur_blanc { "blanc" } else { "noir" },
                    pts,
                    plies
                );
                points += pts;
            }
        }
        let score = points / 10.0;
        println!("mini-arène recherche(3000 nœuds) vs 1 pli : score {score}");
        assert!(
            score >= 0.7,
            "la recherche ne domine pas le réseau brut : {score} < 0.70"
        );
    }

    /// (5) TT persistante : rechercher deux fois la même position sur le même
    /// chercheur → la seconde passe coûte beaucoup moins de nœuds. (Le coup
    /// peut légitimement différer : réutiliser des BORNES mises en cache sous
    /// d'autres fenêtres rend la recherche « instable » entre coups quasi à
    /// égalité — comportement standard de tous les moteurs alpha-bêta à TT.)
    #[test]
    fn tt_reduit_les_noeuds_en_recherche_repetee() {
        let pos = pos_de_fen(
            "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        );
        let mut r = Recherche::new(reseau_reduit(), 16);
        let a = r.cherche(&pos, limites_prof(4));
        let b = r.cherche(&pos, limites_prof(4));
        assert!(a.coup.is_some() && b.coup.is_some());
        // Position calme : les deux scores restent des valeurs réseau.
        assert!(a.score.abs() < 1.0 && b.score.abs() < 1.0);
        assert!(
            b.noeuds * 2 <= a.noeuds,
            "2e recherche pas assez accélérée par la TT : {} vs {}",
            b.noeuds,
            a.noeuds
        );
    }

    /// Fumée avec le réseau COMPLET (Mlp::new(0)) : la recherche s'intègre au
    /// vrai réseau, rend un coup légal et un score réseau borné.
    #[test]
    fn reseau_complet_fumee() {
        let net = Arc::new(Mlp::new(0));
        let mut r = Recherche::new(net, 12);
        let res = r.cherche(&Chess::default(), limites_noeuds(60));
        let coup = res.coup.expect("coup légal en position initiale");
        assert!(Chess::default().is_legal(&coup));
        assert!(res.score.is_finite() && res.score.abs() < 1.0);
        assert!(res.profondeur >= 1);
        assert!(res.noeuds >= 20 && res.noeuds <= 200, "noeuds = {}", res.noeuds);
        assert_eq!(res.scores_racine.len(), 20);
    }

    /// Position terminale : aucun coup → coup None et score de mat/pat.
    #[test]
    fn position_matee_rend_none() {
        // Mat du couloir déjà consommé : trait aux noirs, matés.
        let pos = pos_de_fen("R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1");
        let mut r = Recherche::new(reseau_reduit(), 10);
        let res = r.cherche(&pos, limites_prof(3));
        assert!(res.coup.is_none());
        assert_eq!(res.score, -SCORE_MAT);
        assert!(res.scores_racine.is_empty());
    }

    /// Sonde de performance (ignorée par défaut) : nœuds/s et profondeur en
    /// 100 ms sur la position initiale, réseau complet — à lancer via
    /// `cargo test --lib search:: -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mesure_vitesse() {
        for (nom, net) in [
            ("réseau complet", Arc::new(Mlp::new(0))),
            ("réseau réduit (12 cachés)", reseau_reduit()),
        ] {
            let mut r = Recherche::new(net, 20);
            let debut = Instant::now();
            let res = r.cherche(
                &Chess::default(),
                Limites { max_noeuds: 0, max_profondeur: 0, movetime_ms: 100 },
            );
            let d = debut.elapsed();
            println!(
                "[dev] {nom} : profondeur {} , {} noeuds en {:?} ({:.0} noeuds/s)",
                res.profondeur,
                res.noeuds,
                d,
                res.noeuds as f64 / d.as_secs_f64()
            );
        }
    }
}
