//! Distillation : un réseau élève (architecture élargie, ex. [773,1024,128,1],
//! OU schéma différent via --schema roi8) apprend à IMITER le champion actuel
//! (le prof) — même position → même sortie. On clone la FONCTION d'évaluation,
//! pas la politique : aucune recherche, l'étiquette de chaque position est
//! simplement nn::evalue_position(prof, pos) — le prof peut être de n'importe
//! quel schéma. L'élève ainsi amorcé sert ensuite de point de départ au TD-leaf.
//!
//! C'est AUSSI le transvasement inter-schémas : `--schema roi8` crée un élève
//! au schéma creux roi-zones (`Mlp::new_roi_zones`, entrées `features_roi`),
//! entraîné par `train_batch_actifs` sur les étiquettes du prof (vieux schéma
//! inchangé) — le corpus encode chaque position DANS LE SCHÉMA DE L'ÉLÈVE,
//! l'étiquette vient du prof dans le sien.
//!
//! Options (parse maison sur std::env::args, comme train/serve) :
//!   --teacher models/chess_best.bin   modèle du prof (Mlp::load)
//!   --sizes 773,1024,128,1            architecture de l'élève (entiers, virgules)
//!   --schema classique                schéma de l'élève : "classique" (dense
//!                                     773, défaut) ou "roi8" (creux roi-zones,
//!                                     6149) ; avec roi8, --sizes absent vaut
//!                                     6149,1024,128,1 et sizes[0] DOIT être 6149
//!   --student <chemin>                RE-distillation à chaud : repartir de cet
//!                                     élève EXISTANT (Mlp::load — moments Adam
//!                                     et pas_colonnes conservés tels quels) au
//!                                     lieu d'un réseau neuf ; son schéma doit
//!                                     correspondre à --schema et, si --sizes
//!                                     est fourni, aux tailles du fichier ; son
//!                                     MSE de validation est imprimé AVANT tout
//!                                     entraînement (ligne de base, mêmes
//!                                     20 000 positions que la validation
//!                                     finale)
//!   --positions 600000                taille du corpus de distillation
//!   --epoques-max 6                   plafond d'époques d'apprentissage (≥ 1 ;
//!                                     0 est refusé)
//!   --lr 0.001                        taux d'apprentissage Adam
//!   --seed 0
//!   --out models/distill_student.bin  destination de l'élève
//!
//! Déroulé :
//!   1. corpus : parties de diversification jouées par NetBot(prof, T=0.6, 1 pli)
//!      entrecoupées de coups aléatoires (~1 sur 6) pour couvrir large ; mêmes
//!      règles de nulle qu'en arène (pat, matériel, 50 coups, 3 répétitions,
//!      300 plis max) ; génération parallèle rayon (with_max_len(1)) ;
//!   2. apprentissage : mélange, minibatchs 256, jusqu'à --epoques-max époques
//!      (6 par défaut), arrêt anticipé si la loss moyenne d'une époque passe
//!      sous 0.0004 ;
//!   3. validation sur 20 000 positions FRAÎCHES (autres graines) : MSE
//!      élève-vs-prof et % de positions où |élève - prof| > 0.1 ;
//!   4. sauvegarde vers --out ; code de sortie 0 seulement si MSE < 0.002.

use std::collections::HashMap;
use std::io::Write as _;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, EnPassantMode, Position};

use echec::bots::{Bot, NetBot, RandomBot};
use echec::features::{encode, N_FEATURES};
use echec::features_roi::{self, N_FEATURES_ROI};
use echec::nn::{evalue_position, Mlp, SchemaFeatures};

/// Plis maximum d'une partie de diversification (comme en arène).
const MAX_PLIS: u32 = 300;
/// Température du NetBot prof pendant la génération (1 pli, softmax).
const TEMPERATURE_DIVERSIFICATION: f32 = 0.6;
/// Un coup sur RATIO_ALEA (en moyenne) est joué par le RandomBot.
const RATIO_ALEA: u64 = 6;
/// Taille des minibatchs d'apprentissage.
const MINIBATCH: usize = 256;
/// Nombre maximal d'époques sur le corpus (défaut de --epoques-max).
const MAX_EPOQUES: usize = 6;
/// Arrêt anticipé : loss moyenne d'époque sous ce seuil.
const SEUIL_LOSS_EPOQUE: f32 = 0.0004;
/// Positions fraîches de validation.
const N_VALIDATION: usize = 20_000;
/// Seuil de réussite final : MSE élève-vs-prof.
const SEUIL_MSE_VALIDATION: f32 = 0.002;
/// Estimation grossière de positions par partie (pour dimensionner les vagues
/// de génération ; seule la vitesse en dépend, jamais la correction).
const POSITIONS_PAR_PARTIE_ESTIMEES: usize = 60;
/// Décalage de graines entre corpus d'apprentissage et corpus de validation :
/// les deux ne partagent AUCUNE graine de partie.
const DECALAGE_GRAINES_VALIDATION: u64 = 0x9E37_79B9_7F4A_7C15;

/// Options de la ligne de commande (défauts du contrat).
struct Options {
    teacher: String,
    sizes: Vec<usize>,
    positions: usize,
    lr: f32,
    seed: u64,
    out: String,
    /// Schéma de features de l'ÉLÈVE (le prof garde le sien, lu du fichier).
    schema: SchemaFeatures,
    /// Élève existant à recharger (re-distillation à chaud) ; None = neuf.
    student: Option<String>,
    /// Plafond d'époques d'apprentissage (--epoques-max).
    epoques_max: usize,
    /// --sizes a été fourni explicitement (permet de le confronter au fichier
    /// --student, et à --schema roi8 d'imposer son défaut sinon).
    sizes_explicites: bool,
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

/// Parse une liste d'entiers séparés par des virgules (ex. « 773,1024,128,1 »).
fn parse_tailles(s: &str, nom: &str) -> Vec<usize> {
    s.split(',').map(|morceau| parse_valeur(morceau.trim(), nom)).collect()
}

fn parse_options() -> Options {
    let mut opt = Options {
        teacher: "models/chess_best.bin".to_string(),
        sizes: vec![N_FEATURES, 1024, 128, 1],
        positions: 600_000,
        lr: 0.001,
        seed: 0,
        out: "models/distill_student.bin".to_string(),
        schema: SchemaFeatures::Classique773,
        student: None,
        epoques_max: MAX_EPOQUES,
        sizes_explicites: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--teacher" => opt.teacher = valeur(&args, i, &nom),
            "--sizes" => {
                opt.sizes = parse_tailles(&valeur(&args, i, &nom), &nom);
                opt.sizes_explicites = true;
            }
            "--schema" => {
                opt.schema = match valeur(&args, i, &nom).as_str() {
                    "classique" => SchemaFeatures::Classique773,
                    "roi8" => SchemaFeatures::RoiZones8,
                    autre => {
                        eprintln!(
                            "option --schema : « {autre} » inconnu (attendu : classique | roi8)"
                        );
                        std::process::exit(2);
                    }
                }
            }
            "--student" => opt.student = Some(valeur(&args, i, &nom)),
            "--positions" => opt.positions = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--epoques-max" => {
                opt.epoques_max = parse_valeur(&valeur(&args, i, &nom), &nom);
                // 0 sauterait tout l'apprentissage mais validerait et
                // sauvegarderait quand même : refusé, sûrement involontaire.
                if opt.epoques_max == 0 {
                    eprintln!("option --epoques-max : 0 refuse (au moins une epoque)");
                    std::process::exit(2);
                }
            }
            "--lr" => opt.lr = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--out" => opt.out = valeur(&args, i, &nom),
            autre => {
                eprintln!("option inconnue : {autre}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    // Élève roi-zones sans --sizes : même tête que le défaut historique, mais
    // couche d'entrée au format creux (6149).
    if opt.schema == SchemaFeatures::RoiZones8 && !opt.sizes_explicites {
        opt.sizes = vec![N_FEATURES_ROI, 1024, 128, 1];
    }
    opt
}

fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Corpus de distillation : chaque position est représentée DANS LE SCHÉMA DE
/// L'ÉLÈVE — dense (xs, n × N_FEATURES concaténées) pour `Classique773`,
/// creuse (actifs, une liste d'indices par position) pour `RoiZones8` — et
/// étiquetée par le prof dans SON schéma à lui (`nn::evalue_position`).
/// Un seul des deux champs xs/actifs est rempli, selon le schéma demandé.
struct Corpus {
    xs: Vec<f32>,
    actifs: Vec<Vec<u16>>,
    ys: Vec<f32>,
}

/// Joue UNE partie de diversification et renvoie son corpus. Chaque position
/// visitée (avant le coup) est encodée dans le schéma de l'ÉLÈVE et étiquetée
/// par nn::evalue_position(prof) — pas de recherche. Les coups sont ceux du
/// NetBot(prof, T=0.6, 1 pli), sauf ~1 sur 6 tiré au RandomBot pour élargir la
/// couverture. Mêmes règles de nulle qu'en arène, 300 plis max.
fn partie_corpus(prof: &Mlp, schema_eleve: SchemaFeatures, graine: u64) -> Corpus {
    let mut bot_prof = NetBot::new(prof, graine, TEMPERATURE_DIVERSIFICATION, 1);
    let mut bot_alea = RandomBot::new(graine.wrapping_add(0xA5A5_A5A5));
    // Rng du choix « prof ou aléatoire » : indépendant des deux bots.
    let mut rng = StdRng::seed_from_u64(graine.wrapping_add(0x51ED_51ED));

    let mut pos = Chess::default();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    let mut plis = 0u32;

    // Tampon d'encodage dense de l'élève, et tampon de l'étiquette du prof
    // (utilisé par evalue_position seulement si le prof est dense).
    let mut buf = vec![0.0f32; N_FEATURES];
    let mut tampon_prof: Vec<f32> = Vec::new();
    let mut xs = Vec::new();
    let mut actifs: Vec<Vec<u16>> = Vec::new();
    let mut ys = Vec::new();

    loop {
        // Fins de partie : mat/pat, matériel insuffisant, 50 coups, 300 plis.
        if pos.legal_moves().is_empty()
            || pos.is_insufficient_material()
            || pos.halfmoves() >= 100
            || plis >= MAX_PLIS
        {
            break;
        }

        // Enregistre la position courante : étiquette = prof (son schéma),
        // entrée = encodage au schéma de l'élève.
        ys.push(evalue_position(prof, &pos, &mut tampon_prof));
        match schema_eleve {
            SchemaFeatures::Classique773 => {
                encode(&pos, &mut buf);
                xs.extend_from_slice(&buf);
            }
            SchemaFeatures::RoiZones8 => {
                // ≤ 37 indices (32 pièces + 5 drapeaux).
                let mut a: Vec<u16> = Vec::with_capacity(40);
                features_roi::actifs(&pos, &mut a);
                actifs.push(a);
            }
        }

        // ~1 coup sur 6 : RandomBot ; sinon NetBot du prof (1 pli, T=0.6).
        let coup = if rng.gen_range(0..RATIO_ALEA) == 0 {
            bot_alea.choose(&pos)
        } else {
            bot_prof.choose(&pos)
        }
        .expect("coups légaux non vides");
        pos = pos.play(&coup).expect("coup légal");
        plis += 1;

        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            break; // 3e répétition : nulle
        }
    }
    Corpus { xs, actifs, ys }
}

/// Génère exactement `n_cible` positions étiquetées, par vagues de parties
/// parallèles (rayon, with_max_len(1) : une partie par tâche, sinon les
/// ouvriers libres n'ont plus rien à voler — voir arena.rs). Les graines de
/// partie sont `graine_base + compteur`, déterministes.
fn genere_corpus(
    prof: &Mlp,
    schema_eleve: SchemaFeatures,
    n_cible: usize,
    graine_base: u64,
    etiquette: &str,
) -> Corpus {
    let dense = schema_eleve == SchemaFeatures::Classique773;
    let mut corpus = Corpus {
        xs: Vec::with_capacity(if dense { n_cible * N_FEATURES } else { 0 }),
        actifs: Vec::with_capacity(if dense { 0 } else { n_cible }),
        ys: Vec::with_capacity(n_cible),
    };
    let mut compteur_graines = 0u64;

    while corpus.ys.len() < n_cible {
        let manque = n_cible - corpus.ys.len();
        // Assez de parties pour couvrir le manque (estimation), au moins une
        // vague pleine pour occuper tous les cœurs.
        let n_parties = (manque / POSITIONS_PAR_PARTIE_ESTIMEES + 1).max(rayon::current_num_threads());
        let parties: Vec<Corpus> = (0..n_parties)
            .into_par_iter()
            .with_max_len(1)
            .map(|i| {
                partie_corpus(
                    prof,
                    schema_eleve,
                    graine_base.wrapping_add(compteur_graines + i as u64),
                )
            })
            .collect();
        compteur_graines += n_parties as u64;

        for p in parties {
            if corpus.ys.len() >= n_cible {
                break;
            }
            let prendre = (n_cible - corpus.ys.len()).min(p.ys.len());
            if dense {
                corpus.xs.extend_from_slice(&p.xs[..prendre * N_FEATURES]);
            } else {
                corpus.actifs.extend(p.actifs.into_iter().take(prendre));
            }
            corpus.ys.extend_from_slice(&p.ys[..prendre]);
        }
        print!("\r[{etiquette}] {} / {} positions", corpus.ys.len(), n_cible);
        let _ = std::io::stdout().flush();
    }
    println!();
    corpus
}

/// Passe avant de l'élève sur un corpus de validation (parallèle, lecture
/// seule ; chemin dense ou creux selon le schéma) : renvoie (MSE
/// élève-vs-prof, % de positions à |écart| > 0.1, nombre de ces positions).
fn mesure_validation(eleve: &Mlp, schema: SchemaFeatures, val: &Corpus) -> (f32, f32, usize) {
    let sorties: Vec<f32> = match schema {
        SchemaFeatures::Classique773 => val
            .xs
            .par_chunks(N_FEATURES)
            .map(|x| eleve.forward_one(x))
            .collect(),
        SchemaFeatures::RoiZones8 => val
            .actifs
            .par_iter()
            .map(|a| eleve.forward_actifs(a))
            .collect(),
    };
    let mut somme_carres = 0.0f64;
    let mut n_gros_ecarts = 0usize;
    for (s, t) in sorties.iter().zip(&val.ys) {
        let e = (s - t) as f64;
        somme_carres += e * e;
        if (s - t).abs() > 0.1 {
            n_gros_ecarts += 1;
        }
    }
    let mse = (somme_carres / val.ys.len() as f64) as f32;
    let pct_gros_ecarts = 100.0 * n_gros_ecarts as f32 / val.ys.len() as f32;
    (mse, pct_gros_ecarts, n_gros_ecarts)
}

fn main() {
    echec::pleine_puissance();
    let debut = Instant::now();
    let opt = parse_options();

    // Garde d'architecture : la sortie doit être scalaire (tanh). Vérifié ICI,
    // avant toute génération de corpus, avec le même code de sortie 2 que les
    // autres erreurs d'options — plus lisible que la panique du garde de
    // `Mlp::new_avec_tailles` pour un --sizes tronqué (ex. « 773,1024,128 »).
    if opt.sizes.last() != Some(&1) {
        eprintln!(
            "option --sizes : la dernière couche doit valoir 1 (sortie scalaire tanh), reçu {:?}",
            opt.sizes
        );
        std::process::exit(2);
    }
    // Même politique pour la couche d'entrée d'un élève roi-zones : sortie
    // propre AVANT la génération du corpus, plutôt que la panique du garde de
    // `Mlp::new_roi_zones` après coup.
    if opt.schema == SchemaFeatures::RoiZones8 && opt.sizes.first() != Some(&N_FEATURES_ROI) {
        eprintln!(
            "option --sizes : avec --schema roi8, la couche d'entrée doit faire \
             N_FEATURES_ROI = {N_FEATURES_ROI}, reçu {:?}",
            opt.sizes
        );
        std::process::exit(2);
    }

    // 1. Prof et élève. Le prof garde SON schéma (lu du fichier, dense 773
    //    historique ou roi-zones) ; l'élève est créé au schéma demandé.
    let prof = Mlp::load(&opt.teacher).unwrap_or_else(|e| {
        eprintln!("chargement du prof {} : {e}", opt.teacher);
        std::process::exit(2);
    });
    // Élève : soit rechargé tel quel (re-distillation à chaud, --student —
    // poids, moments Adam et pas_colonnes du fichier), soit créé neuf.
    let mut eleve = match &opt.student {
        Some(chemin) => {
            let charge = Mlp::load(chemin).unwrap_or_else(|e| {
                eprintln!("chargement de l'eleve {chemin} : {e}");
                std::process::exit(2);
            });
            if charge.schema() != opt.schema {
                eprintln!(
                    "option --student : {chemin} est au schema {:?}, mais --schema demande {:?}",
                    charge.schema(),
                    opt.schema
                );
                std::process::exit(2);
            }
            if opt.sizes_explicites && charge.sizes != opt.sizes {
                eprintln!(
                    "option --student : {chemin} a les tailles {:?}, mais --sizes demande {:?}",
                    charge.sizes, opt.sizes
                );
                std::process::exit(2);
            }
            charge
        }
        None => match opt.schema {
            SchemaFeatures::Classique773 => Mlp::new_avec_tailles(&opt.sizes, opt.seed),
            SchemaFeatures::RoiZones8 => Mlp::new_roi_zones(&opt.sizes, opt.seed),
        },
    };
    println!(
        "distillation : prof {} {:?} (schema {:?}) -> eleve {:?} (schema {:?}) | {} positions | lr {} | graine {}",
        opt.teacher, prof.sizes, prof.schema(), eleve.sizes, eleve.schema(),
        opt.positions, opt.lr, opt.seed
    );

    // Ligne de base (--student) : MSE de validation de l'élève chargé, AVANT
    // tout entraînement, sur les MÊMES 20 000 positions fraîches que la
    // validation finale (graines identiques → corpus identique, conservé pour
    // l'étape 4 plutôt que régénéré).
    let mut val_precalculee: Option<Corpus> = None;
    if opt.student.is_some() {
        let val = genere_corpus(
            &prof,
            opt.schema,
            N_VALIDATION,
            opt.seed.wrapping_add(DECALAGE_GRAINES_VALIDATION),
            "validation",
        );
        let (mse0, pct0, n0) = mesure_validation(&eleve, opt.schema, &val);
        println!(
            "ligne de base (eleve charge) : MSE eleve-vs-prof {mse0:.6} | |ecart| > 0.1 : {pct0:.2} % ({n0} / {})",
            val.ys.len()
        );
        val_precalculee = Some(val);
    }

    // 2. Corpus d'apprentissage (au schéma de l'élève).
    let corpus = genere_corpus(&prof, opt.schema, opt.positions, opt.seed, "corpus");

    // 3. Apprentissage : mélange des indices, minibatchs de 256, jusqu'à
    //    --epoques-max époques (6 par défaut), arrêt anticipé si la loss
    //    moyenne d'époque < 0.0004.
    //    Chemin DENSE (train_batch) pour Classique773, chemin CREUX
    //    (train_batch_actifs) pour RoiZones8 — mêmes hyperparamètres.
    let n = corpus.ys.len();
    let mut indices: Vec<usize> = (0..n).collect();
    // Rng du mélange : décalé de la graine des parties (aucun recouvrement).
    let mut rng = StdRng::seed_from_u64(opt.seed.wrapping_add(0x4D45_4C41_4E47_45u64));
    let mut lot_xs: Vec<f32> = Vec::with_capacity(MINIBATCH * N_FEATURES);
    let mut lot_ys: Vec<f32> = Vec::with_capacity(MINIBATCH);
    let mut epoques_faites = 0usize;
    for epoque in 1..=opt.epoques_max {
        indices.shuffle(&mut rng);
        let mut somme_loss = 0.0f64;
        let mut n_batchs = 0usize;
        for lot in indices.chunks(MINIBATCH) {
            let loss = match opt.schema {
                SchemaFeatures::Classique773 => {
                    lot_xs.clear();
                    lot_ys.clear();
                    for &i in lot {
                        lot_xs.extend_from_slice(
                            &corpus.xs[i * N_FEATURES..(i + 1) * N_FEATURES],
                        );
                        lot_ys.push(corpus.ys[i]);
                    }
                    eleve.train_batch(&lot_xs, &lot_ys, opt.lr)
                }
                SchemaFeatures::RoiZones8 => {
                    let lots: Vec<(Vec<u16>, f32)> = lot
                        .iter()
                        .map(|&i| (corpus.actifs[i].clone(), corpus.ys[i]))
                        .collect();
                    eleve.train_batch_actifs(&lots, opt.lr)
                }
            };
            somme_loss += loss as f64;
            n_batchs += 1;
            if n_batchs % 200 == 0 {
                print!(
                    "\r[epoque {epoque}] batch {n_batchs} | loss moyenne {:.6}",
                    somme_loss / n_batchs as f64
                );
                let _ = std::io::stdout().flush();
            }
        }
        let loss_epoque = (somme_loss / n_batchs.max(1) as f64) as f32;
        epoques_faites = epoque;
        println!("\r[epoque {epoque}] {n_batchs} batchs | loss moyenne {loss_epoque:.6}");
        if loss_epoque < SEUIL_LOSS_EPOQUE {
            println!("arret anticipe : loss d'epoque {loss_epoque:.6} < {SEUIL_LOSS_EPOQUE}");
            break;
        }
    }
    drop(corpus);

    // 4. Validation : positions FRAÎCHES (graines décalées, jamais utilisées
    //    par le corpus d'apprentissage) — celles de la ligne de base si elle a
    //    eu lieu (mêmes graines), générées maintenant sinon.
    let val = val_precalculee.take().unwrap_or_else(|| {
        genere_corpus(
            &prof,
            opt.schema,
            N_VALIDATION,
            opt.seed.wrapping_add(DECALAGE_GRAINES_VALIDATION),
            "validation",
        )
    });
    let (mse, pct_gros_ecarts, n_gros_ecarts) = mesure_validation(&eleve, opt.schema, &val);
    println!(
        "validation : MSE eleve-vs-prof {mse:.6} | |ecart| > 0.1 : {pct_gros_ecarts:.2} % ({n_gros_ecarts} / {})",
        val.ys.len()
    );

    // Sauvegarde de l'élève.
    if let Err(e) = eleve.save(&opt.out) {
        eprintln!("sauvegarde de {} : {e}", opt.out);
        std::process::exit(2);
    }
    println!("eleve sauvegarde -> {}", opt.out);

    // 5. Récapitulatif.
    let duree = debut.elapsed().as_secs_f64();
    println!(
        "recap : tailles {:?} | {} positions | {epoques_faites} epoque(s) | MSE finale {mse:.6} | {duree:.1} s",
        eleve.sizes, opt.positions
    );

    // Code de sortie : 0 seulement si la distillation est fidèle.
    std::process::exit(if mse < SEUIL_MSE_VALIDATION { 0 } else { 1 });
}
