//! Calibration oracle : mesure, au voisinage du champion, la géométrie de la
//! loss MSE vers les étiquettes Stockfish — pour départager trois hypothèses
//! sur l'anti-corrélation objectif/force observée (greffe roi8) :
//! décalage d'ÉCHELLE (la recherche n'utilise que l'ordre, la MSE exige des
//! valeurs), BRUIT des étiquettes à movetime court, DÉSACCORD réel.
//!
//! `calibration --net <chemin> --oracle <chemin_stockfish> --movetime 40
//!  --positions 6000 --seed S --out <csv>`
//!
//! Déroulé :
//! 1. positions par self-play du réseau --net, MÊME distribution que
//!    l'entraînement (régime recherche : mêmes constantes OptionsRecherche —
//!    températures 0.2 / ouverture 0.8, arbitrage 0.92 × 4 plis — départs
//!    livre/finales comme relance_train.ps1, arbitrage sur l'étiquette oracle
//!    à poids_prof 1.0, repli élève si le moteur ne répond pas) ;
//! 2. par position : v_champion = nn::evalue_position (perspective du trait),
//!    et DEUX étiquettes oracle indépendantes par LE MÊME CODE que le régime
//!    --oracle de l'entraînement (UciEngine::evalue_fen sur la FEN
//!    EnPassantMode::Legal : tanh(cp/300), mats → ±1, score du point de vue
//!    du trait, aucun renversement de signe — voir uci.rs et selfplay.rs) —
//!    label1 pendant la partie (TT du moteur chauffée au fil de la partie,
//!    comme à l'entraînement), label2 en REJOUANT la même séquence de
//!    requêtes sur un moteur repassé par ucinewgame (mêmes conditions,
//!    seconde réalisation du même processus d'étiquetage) ;
//! 3. CSV v_champion,label1,label2 ;
//! 4. analyse : corrélations Pearson/Spearman, table de calibration en
//!    20 quantiles, décomposition de la loss (bruit / échelle / désaccord)
//!    par régression isotone croisée, garde-fou sur deux moitiés.
//!
//! Mode AJUSTEMENT (`--fit <sortie.tsv>`) : ajuste une régression isotone
//! croissante (PAV) label → v sur un CSV v,label1,label2 existant
//! (`--from-csv <chemin>` : aucune mesure, aucun moteur) OU sur les triplets
//! du run courant, la réduit en ~64 nœuds aux quantiles du label, borne la
//! table à -1/+1 par prolongement des segments extrêmes, et écrit un TSV
//! « label<TAB>v » aux deux colonnes STRICTEMENT croissantes — le format lu
//! par train.exe --recalibrage (voir selfplay::Recalibrage).

use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::zobrist::{Zobrist64, ZobristHash};
use shakmaty::{Chess, Color, EnPassantMode, Position};

use echec::bots::echantillonne_scores_racine;
use echec::nn::{evalue_position, Mlp};
use echec::search;
use echec::selfplay::OptionsRecherche;
use echec::uci::UciEngine;

/// Plis maximum d'une partie (même valeur que MAX_PLIES de train.rs).
const MAX_PLIES: u32 = 300;
/// Taille de TT des chercheurs de self-play (même valeur que train.rs).
const TAILLE_TT_LOG2: u32 = 18;
/// Estimation grossière de positions par partie (dimensionne les vagues ;
/// seule la vitesse en dépend, jamais la correction).
const POSITIONS_PAR_PARTIE_ESTIMEES: usize = 60;

struct Options {
    net: String,
    oracle: String,
    movetime: u32,
    positions: usize,
    seed: u64,
    out: String,
    /// Nœuds de recherche par coup du self-play (défaut : régime courant de
    /// relance_train.ps1).
    search_nodes: u64,
    /// Proportions de départs variés (défauts : régime courant).
    departs_ouvertures: f32,
    departs_finales: f32,
    departs_transition: f32,
    /// Threads rayon (0 = défaut rayon).
    threads: usize,
    /// Ajoute au CSV existant au lieu de l'écraser (tranches : lancer
    /// plusieurs runs avec des graines différentes ; l'analyse imprimée ne
    /// porte que sur les triplets du run courant).
    append: bool,
    /// Mode ajustement : chemin du TSV « label<TAB>v » à écrire (vide =
    /// désactivé). Avec --from-csv : aucune mesure, ajustement seul.
    fit: String,
    /// CSV v,label1,label2 existant servant d'entrée au mode --fit.
    from_csv: String,
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
        net: "models/chess_best.bin".to_string(),
        oracle: "engines/stockfish/stockfish-windows-x86-64-avx2.exe".to_string(),
        movetime: 40,
        positions: 6000,
        seed: 0,
        out: "calibration.csv".to_string(),
        search_nodes: 8000,
        departs_ouvertures: 0.6,
        departs_finales: 0.2,
        departs_transition: 0.0,
        threads: 0,
        append: false,
        fit: String::new(),
        from_csv: String::new(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--net" => opt.net = valeur(&args, i, &nom),
            "--oracle" => opt.oracle = valeur(&args, i, &nom),
            "--movetime" => opt.movetime = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--positions" => opt.positions = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--out" => opt.out = valeur(&args, i, &nom),
            "--search-nodes" => opt.search_nodes = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--departs-ouvertures" => {
                opt.departs_ouvertures = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--departs-finales" => {
                opt.departs_finales = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--departs-transition" => {
                opt.departs_transition = parse_valeur(&valeur(&args, i, &nom), &nom)
            }
            "--threads" => opt.threads = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--append" => {
                opt.append = true;
                i += 1;
                continue; // drapeau sans valeur
            }
            "--fit" => opt.fit = valeur(&args, i, &nom),
            "--from-csv" => opt.from_csv = valeur(&args, i, &nom),
            autre => {
                eprintln!("option inconnue : {autre}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    // Garde-fou des parts (même refus que train.rs) : une troncature
    // silencieuse du tirage fausserait la mesure sans prévenir.
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

/// Mélangeur déterministe (copie de train.rs : mêmes constantes, pour que le
/// tirage des départs suive la même dérivation de graines que l'entraînement).
fn derive_graine(base: u64, sel: u64) -> u64 {
    let mut z = base ^ sel.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn zobrist(pos: &Chess) -> u64 {
    let h: Zobrist64 = pos.zobrist_hash(EnPassantMode::Legal);
    h.0
}

/// Une position mesurée : FEN (mode Legal, la même que celle envoyée à
/// l'oracle), évaluation nue du champion (perspective du trait), première
/// étiquette oracle (None : moteur muet pour cette position → ligne écartée).
struct Ligne {
    fen: String,
    v_champion: f32,
    label1: Option<f32>,
}

/// Emprunte un moteur au pool (santé vérifiée par isready), ou en relance un.
/// None : relance impossible — la partie se joue sans oracle (repli élève,
/// lignes inutilisables, comptées et écartées).
fn emprunte_moteur(
    pool: &Mutex<Vec<UciEngine>>,
    chemin: &str,
    movetime: u32,
) -> Option<UciEngine> {
    let emprunte = pool.lock().unwrap_or_else(|e| e.into_inner()).pop();
    let vivant = match emprunte {
        Some(mut m) => m.pret().is_ok().then_some(m),
        None => None,
    };
    vivant.or_else(|| UciEngine::lance_pleine_force(chemin, movetime).ok())
}

/// Joue UNE partie de self-play dans les conditions de l'entraînement (régime
/// recherche + oracle, poids_prof 1.0) et renvoie ses positions mesurées.
/// Miroir de selfplay::partie_recherche_interne (mêmes constantes
/// OptionsRecherche, mêmes règles de fin, même arbitrage, même point
/// d'enregistrement AVANT le coup) — recopié ici car il faut conserver les
/// POSITIONS, que GameRecord ne restitue pas. L'étiquette label1 est calculée
/// par LE MÊME CODE que l'entraînement : UciEngine::evalue_fen sur la FEN
/// EnPassantMode::Legal (voir selfplay.rs, branche Etiqueteur::Oracle).
fn joue_partie(
    net: &Arc<Mlp>,
    opt: &Options,
    opts: &OptionsRecherche,
    pool: &Mutex<Vec<UciEngine>>,
    graine: u64,
) -> Vec<Ligne> {
    let mut chercheur = search::Recherche::new(net.clone(), TAILLE_TT_LOG2);
    chercheur.nouvelle_partie();
    // Départ varié : même dérivation de graine et même tirage que train.rs
    // (tirage_complet, part de transition comprise — part nulle = tirage
    // historique, bit à bit).
    let depart = {
        let mut rng_depart = StdRng::seed_from_u64(derive_graine(graine, 0xDE9A47));
        echec::departs::tirage_complet(
            &mut rng_depart,
            opt.departs_ouvertures,
            opt.departs_finales,
            opt.departs_transition,
        )
    };
    let mut pos = depart.pos.clone();
    let plis_chauds = depart.plis_chauds;

    let mut moteur = emprunte_moteur(pool, &opt.oracle, opt.movetime);
    if let Some(o) = moteur.as_mut() {
        // Comme à l'entraînement : un ucinewgame par partie (un échec n'est
        // pas fatal, evalue_fen renverra None et la ligne sera écartée).
        let _ = o.nouvelle_partie();
    }

    let mut rng = StdRng::seed_from_u64(graine);
    let limites = search::Limites {
        max_noeuds: opts.nodes_par_coup,
        max_profondeur: 0,
        movetime_ms: 0,
    };

    let mut lignes: Vec<Ligne> = Vec::new();
    let mut tampon: Vec<f32> = Vec::new();
    let mut repetitions: HashMap<u64, u8> = HashMap::new();
    repetitions.insert(zobrist(&pos), 1);
    let mut serie_arbitrage: i32 = 0;
    let mut plies = 0u32;

    loop {
        let coups = pos.legal_moves();
        if coups.is_empty()
            || pos.is_insufficient_material()
            || pos.halfmoves() >= 100
            || plies >= opts.max_plies
        {
            break;
        }

        let res = chercheur.cherche(&pos, limites);
        let v_eleve = res.score.clamp(-1.0, 1.0);

        // MÊME CODE d'étiquetage que l'entraînement (selfplay.rs,
        // Etiqueteur::Oracle) : FEN mode Legal → evalue_fen (tanh(cp/300),
        // mats ±1, point de vue du trait, aucun renversement de signe).
        let fen = Fen::from_position(pos.clone(), EnPassantMode::Legal).to_string();
        let label1 = moteur.as_mut().and_then(|o| o.evalue_fen(&fen));
        // poids_prof 1.0 (régime courant) : l'arbitrage suit l'oracle,
        // repli élève si le moteur est muet — comme à l'entraînement.
        let v_racine = label1.unwrap_or(v_eleve);

        // Enregistrement AVANT le coup, comme selfplay::enregistre_position.
        lignes.push(Ligne {
            fen,
            v_champion: evalue_position(net, &pos, &mut tampon),
            label1,
        });

        // Arbitrage : v_racine est du point de vue du trait, qui alterne.
        let v_blancs = if pos.turn() == Color::White { v_racine } else { -v_racine };
        if v_blancs >= opts.seuil_arbitrage {
            serie_arbitrage = if serie_arbitrage >= 0 { serie_arbitrage + 1 } else { 1 };
        } else if v_blancs <= -opts.seuil_arbitrage {
            serie_arbitrage = if serie_arbitrage <= 0 { serie_arbitrage - 1 } else { -1 };
        } else {
            serie_arbitrage = 0;
        }
        if opts.plis_arbitrage > 0 && serie_arbitrage.unsigned_abs() >= opts.plis_arbitrage {
            break;
        }

        // Ouverture diversifiée (plis chauds du départ), puis régime normal.
        let t = if plies < plis_chauds {
            opts.temperature_ouverture
        } else {
            opts.temperature
        };
        let m = if t > 0.0 {
            echantillonne_scores_racine(&res.scores_racine, t, &mut rng)
                .or(res.coup)
                .expect("coups légaux non vides")
        } else {
            res.coup.expect("coups légaux non vides")
        };
        pos = pos.play(&m).expect("coup légal");
        plies += 1;

        let compteur = repetitions.entry(zobrist(&pos)).or_insert(0);
        *compteur += 1;
        if *compteur >= 3 {
            break;
        }
    }

    if let Some(m) = moteur {
        pool.lock().unwrap_or_else(|e| e.into_inner()).push(m);
    }
    lignes
}

/// Deuxième passe d'étiquetage d'UNE partie : rejoue la MÊME séquence de
/// requêtes (ucinewgame puis evalue_fen position par position, dans l'ordre)
/// sur un moteur du pool — seconde réalisation indépendante du processus
/// d'étiquetage de l'entraînement, TT chauffée de la même façon.
fn etiquette_partie_bis(
    partie: &[Ligne],
    opt: &Options,
    pool: &Mutex<Vec<UciEngine>>,
) -> Vec<Option<f32>> {
    let mut moteur = emprunte_moteur(pool, &opt.oracle, opt.movetime);
    if let Some(o) = moteur.as_mut() {
        let _ = o.nouvelle_partie();
    }
    let labels: Vec<Option<f32>> = partie
        .iter()
        .map(|l| moteur.as_mut().and_then(|o| o.evalue_fen(&l.fen)))
        .collect();
    if let Some(m) = moteur {
        pool.lock().unwrap_or_else(|e| e.into_inner()).push(m);
    }
    labels
}

// ---------------------------------------------------------------------------
// Statistiques
// ---------------------------------------------------------------------------

fn moyenne(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Corrélation de Pearson.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (moyenne(a), moyenne(b));
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa.sqrt() * sbb.sqrt()).max(1e-12)
}

/// Rangs (1..n) avec rangs moyens sur les ex æquo (les mats à ±1 en créent).
fn rangs(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("valeurs finies"));
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let moyen = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            r[idx[k]] = moyen;
        }
        i = j + 1;
    }
    r
}

/// Corrélation de Spearman = Pearson des rangs.
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&rangs(a), &rangs(b))
}

/// Régression isotone CROISSANTE x → y par PAV (pool adjacent violators) :
/// fonction en escalier, valeur du palier = moyenne du bloc fusionné.
struct Isotone {
    /// Abscisses triées DISTINCTES (ex æquo agrégés à l'ajustement).
    xs: Vec<f64>,
    /// Valeur ajustée de l'abscisse correspondante (croissante par construction).
    valeurs: Vec<f64>,
}

fn ajuste_isotone(x: &[f64], y: &[f64]) -> Isotone {
    let mut idx: Vec<usize> = (0..x.len()).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).expect("valeurs finies"));
    // Agrégation des ex æquo AVANT le PAV : une abscisse distincte = un point
    // pondéré (somme des y, effectif). Sans cela, une sous-suite croissante
    // interne aux ex æquo n'est jamais fusionnée et g prend plusieurs valeurs
    // à la même abscisse (faux notamment aux mats, empilés à ±1).
    let mut xs: Vec<f64> = Vec::new();
    let mut points: Vec<(f64, usize)> = Vec::new();
    for &i in &idx {
        if xs.last() == Some(&x[i]) {
            let dernier = points.last_mut().expect("points non vide");
            dernier.0 += y[i];
            dernier.1 += 1;
        } else {
            xs.push(x[i]);
            points.push((y[i], 1));
        }
    }
    // Pile de blocs (somme des y, effectif, nb d'abscisses) ; fusion tant que
    // les moyennes décroissent — l'invariant final est une suite croissante.
    let mut blocs: Vec<(f64, usize, usize)> = Vec::with_capacity(points.len());
    for &(somme, effectif) in &points {
        blocs.push((somme, effectif, 1));
        while blocs.len() >= 2 {
            let dernier = blocs[blocs.len() - 1];
            let avant = blocs[blocs.len() - 2];
            if avant.0 / avant.1 as f64 > dernier.0 / dernier.1 as f64 {
                blocs.pop();
                blocs.pop();
                blocs.push((avant.0 + dernier.0, avant.1 + dernier.1, avant.2 + dernier.2));
            } else {
                break;
            }
        }
    }
    let mut valeurs = Vec::with_capacity(xs.len());
    for (somme, effectif, nb_abscisses) in blocs {
        let m = somme / effectif as f64;
        valeurs.extend(std::iter::repeat(m).take(nb_abscisses));
    }
    Isotone { xs, valeurs }
}

impl Isotone {
    /// Valeur au palier du plus grand x d'ajustement <= q (bornes clampées).
    fn evalue(&self, q: f64) -> f64 {
        let i = self.xs.partition_point(|&x| x <= q);
        if i == 0 {
            self.valeurs[0]
        } else {
            self.valeurs[i - 1]
        }
    }
}

/// Décomposition de la loss sur des triplets (v_champion, label1, label2).
struct Decomposition {
    n: usize,
    mse_total: f64,
    bruit: f64,
    mse_apres_g: f64,
    echelle: f64,
    desaccord: f64,
}

/// MSE_total = E[(v - label1)²] ; bruit = Var(label1 - label2)/2 ;
/// g = isotone label → v ajustée sur (label2, v) et évaluée sur label1
/// (validation croisée : g ne voit jamais label1) ;
/// échelle = MSE_total - MSE_après_g (ce qu'une recalibration monotone
/// récupérerait) ; désaccord = MSE_après_g - bruit (résidu net du bruit).
/// Les trois parts somment à MSE_total.
fn decompose(triplets: &[(f64, f64, f64)]) -> Decomposition {
    let n = triplets.len();
    let v: Vec<f64> = triplets.iter().map(|t| t.0).collect();
    let l1: Vec<f64> = triplets.iter().map(|t| t.1).collect();
    let l2: Vec<f64> = triplets.iter().map(|t| t.2).collect();

    let mse_total = moyenne(&v.iter().zip(&l1).map(|(a, b)| (a - b) * (a - b)).collect::<Vec<_>>());
    let d: Vec<f64> = l1.iter().zip(&l2).map(|(a, b)| a - b).collect();
    let md = moyenne(&d);
    let bruit = moyenne(&d.iter().map(|x| (x - md) * (x - md)).collect::<Vec<_>>()) / 2.0;

    let g = ajuste_isotone(&l2, &v);
    let mse_apres_g = moyenne(
        &v.iter().zip(&l1).map(|(a, b)| (a - g.evalue(*b)) * (a - g.evalue(*b))).collect::<Vec<_>>(),
    );

    Decomposition {
        n,
        mse_total,
        bruit,
        mse_apres_g,
        echelle: mse_total - mse_apres_g,
        desaccord: mse_apres_g - bruit,
    }
}

fn imprime_decomposition(titre: &str, d: &Decomposition) {
    let pct = |x: f64| 100.0 * x / d.mse_total.max(1e-12);
    println!("{titre} (n = {}) :", d.n);
    println!("  MSE_total                   = {:.6}", d.mse_total);
    println!("  MSE_apres_g (isotone l->v)  = {:.6}", d.mse_apres_g);
    println!(
        "  part ECHELLE   = MSE_total - MSE_apres_g = {:.6}  ({:.1} %)",
        d.echelle,
        pct(d.echelle)
    );
    println!(
        "  part BRUIT     = Var(l1 - l2)/2          = {:.6}  ({:.1} %)",
        d.bruit,
        pct(d.bruit)
    );
    println!(
        "  part DESACCORD = MSE_apres_g - bruit     = {:.6}  ({:.1} %)",
        d.desaccord,
        pct(d.desaccord)
    );
}

// ---------------------------------------------------------------------------
// Mode --fit : table de recalibrage label → v pour train.exe --recalibrage
// ---------------------------------------------------------------------------

/// Nombre visé de nœuds de la table réduite (quantiles du label).
const N_NOEUDS_FIT: usize = 64;
/// Écart minimal imposé entre deux v consécutifs (les paliers PAV sont plats ;
/// train.exe exige une croissance STRICTE des deux colonnes).
const EPS_STRICT: f64 = 1e-6;

/// Lit un CSV « v_champion,label1,label2 » (entête tolérée) en triplets.
fn lit_csv_triplets(chemin: &str) -> Vec<(f64, f64, f64)> {
    let texte = std::fs::read_to_string(chemin).unwrap_or_else(|e| {
        eprintln!("--from-csv {chemin} : lecture impossible ({e})");
        std::process::exit(2);
    });
    let mut triplets = Vec::new();
    for (i, ligne) in texte.lines().enumerate() {
        let l = ligne.trim();
        if l.is_empty() || l.starts_with("v_champion") {
            continue; // entête ou ligne vide
        }
        let champs: Vec<&str> = l.split(',').collect();
        let valeurs: Option<Vec<f64>> =
            champs.iter().map(|c| c.trim().parse().ok()).collect();
        match valeurs.as_deref() {
            Some([v, l1, l2]) => triplets.push((*v, *l1, *l2)),
            _ => {
                eprintln!("--from-csv {chemin} ligne {} : « {l} » imparsable", i + 1);
                std::process::exit(2);
            }
        }
    }
    if triplets.is_empty() {
        eprintln!("--from-csv {chemin} : aucun triplet");
        std::process::exit(2);
    }
    triplets
}

/// Ajuste g (isotone croissante label → v, PAV) sur les triplets — les DEUX
/// étiquettes sont empilées, appariées au même v (deux réalisations du même
/// processus d'étiquetage : le bruit se moyenne) —, réduit en ~N_NOEUDS_FIT
/// nœuds aux quantiles du label, borne à -1/+1 par prolongement des segments
/// extrêmes (v clampé à [-1, 1]), impose la croissance STRICTE des deux
/// colonnes (paliers PAV décollés d'EPS_STRICT), écrit le TSV « label<TAB>v »
/// et imprime la table. Vérifie monotonie et bornes avant d'écrire.
fn ajuste_et_ecrit_table(triplets: &[(f64, f64, f64)], chemin: &str) {
    // Empilement (label, v) des deux étiquettes.
    let mut labels: Vec<f64> = Vec::with_capacity(triplets.len() * 2);
    let mut vs: Vec<f64> = Vec::with_capacity(triplets.len() * 2);
    for &(v, l1, l2) in triplets {
        labels.push(l1);
        vs.push(v);
        labels.push(l2);
        vs.push(v);
    }
    let g = ajuste_isotone(&labels, &vs);

    // Quantiles du label (dédupliqués), évalués sur g.
    let mut tri = labels.clone();
    tri.sort_by(|a, b| a.partial_cmp(b).expect("valeurs finies"));
    let mut noeuds: Vec<(f64, f64)> = Vec::new();
    for k in 0..N_NOEUDS_FIT {
        let q = tri[k * (tri.len() - 1) / (N_NOEUDS_FIT - 1)];
        if noeuds.last().map(|&(x, _)| q > x).unwrap_or(true) {
            noeuds.push((q, g.evalue(q)));
        }
    }
    // Croissance stricte de v : les paliers PAV sont décollés d'EPS_STRICT
    // (l'interpolation linéaire de train.exe exige deux colonnes strictement
    // croissantes ; l'écart est invisible à l'échelle des cibles).
    for k in 1..noeuds.len() {
        if noeuds[k].1 <= noeuds[k - 1].1 {
            noeuds[k].1 = noeuds[k - 1].1 + EPS_STRICT;
        }
    }
    // Bornes -1/+1 : prolongement linéaire des segments extrêmes, v clampé à
    // [-1, 1] et maintenu strictement ordonné.
    if noeuds.len() < 2 {
        eprintln!("--fit : moins de 2 noeuds distincts, table inutilisable");
        std::process::exit(1);
    }
    if noeuds[0].0 > -1.0 {
        let (x0, y0) = noeuds[0];
        let (x1, y1) = noeuds[1];
        let pente = (y1 - y0) / (x1 - x0);
        let v = (y0 + pente * (-1.0 - x0)).clamp(-1.0, y0 - EPS_STRICT);
        noeuds.insert(0, (-1.0, v));
    }
    if noeuds[noeuds.len() - 1].0 < 1.0 {
        let (x0, y0) = noeuds[noeuds.len() - 2];
        let (x1, y1) = noeuds[noeuds.len() - 1];
        let pente = (y1 - y0) / (x1 - x0);
        let v = (y1 + pente * (1.0 - x1)).clamp(y1 + EPS_STRICT, 1.0);
        noeuds.push((1.0, v));
    }

    // Vérification finale : bornes exactes et croissance stricte des deux
    // colonnes — le contrat du fichier, revérifié par train.exe au chargement.
    assert_eq!(noeuds[0].0, -1.0, "premiere borne != -1");
    assert_eq!(noeuds[noeuds.len() - 1].0, 1.0, "derniere borne != +1");
    for k in 1..noeuds.len() {
        assert!(
            noeuds[k].0 > noeuds[k - 1].0 && noeuds[k].1 > noeuds[k - 1].1,
            "monotonie stricte violee au noeud {k} : {:?} -> {:?}",
            noeuds[k - 1],
            noeuds[k]
        );
        assert!(noeuds[k].1.abs() <= 1.0, "v hors [-1, 1] au noeud {k}");
    }

    let mut contenu = String::new();
    for &(x, y) in &noeuds {
        contenu.push_str(&format!("{x:.6}\t{y:.6}\n"));
    }
    std::fs::write(chemin, &contenu).unwrap_or_else(|e| {
        eprintln!("--fit {chemin} : ecriture impossible ({e})");
        std::process::exit(2);
    });

    println!(
        "table de recalibrage : {} noeuds -> {chemin} ({} triplets, \
         2 etiquettes empilees)",
        noeuds.len(),
        triplets.len()
    );
    println!("{:>4} {:>12} {:>12}", "k", "label", "v");
    for (k, &(x, y)) in noeuds.iter().enumerate() {
        println!("{:>4} {:>12.6} {:>12.6}", k + 1, x, y);
    }
    println!("monotonie stricte et bornes -1/+1 : verifiees");
}

fn main() {
    echec::pleine_puissance();
    let debut = Instant::now();
    let opt = parse_options();
    if !opt.from_csv.is_empty() && opt.fit.is_empty() {
        eprintln!("--from-csv sans --fit : rien a faire");
        std::process::exit(2);
    }
    // Mode ajustement pur : CSV existant → table, sans mesure ni moteur.
    if !opt.fit.is_empty() && !opt.from_csv.is_empty() {
        let triplets = lit_csv_triplets(&opt.from_csv);
        ajuste_et_ecrit_table(&triplets, &opt.fit);
        println!("duree totale : {:.1} s", debut.elapsed().as_secs_f64());
        return;
    }
    if opt.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(opt.threads)
            .build_global()
            .expect("pool rayon");
    }

    let net = Arc::new(Mlp::load(&opt.net).unwrap_or_else(|e| {
        eprintln!("chargement du reseau {} : {e}", opt.net);
        std::process::exit(2);
    }));
    println!(
        "calibration : net {} {:?} (schema {:?}) | oracle {} movetime {} ms | {} positions | \
         {} noeuds/coup | departs {}/{} | graine {}",
        opt.net,
        net.sizes,
        net.schema(),
        opt.oracle,
        opt.movetime,
        opt.positions,
        opt.search_nodes,
        opt.departs_ouvertures,
        opt.departs_finales,
        opt.seed
    );

    // Constantes du régime recherche de l'entraînement (températures,
    // arbitrage) : les défauts FIGÉS d'OptionsRecherche, comme train.rs.
    let opts_recherche = OptionsRecherche {
        nodes_par_coup: opt.search_nodes,
        max_plies: MAX_PLIES,
        ..Default::default()
    };

    // Pool de moteurs pleine force, un par thread (même schéma que train.rs).
    let n_moteurs = rayon::current_num_threads();
    let mut moteurs = Vec::with_capacity(n_moteurs);
    for _ in 0..n_moteurs {
        match UciEngine::lance_pleine_force(&opt.oracle, opt.movetime) {
            Ok(m) => moteurs.push(m),
            Err(e) => {
                eprintln!("--oracle {} : lancement impossible ({e})", opt.oracle);
                std::process::exit(2);
            }
        }
    }
    let pool = Mutex::new(moteurs);

    // 1. Self-play par vagues jusqu'à couvrir --positions (lignes utilisables,
    //    c'est-à-dire avec label1), puis troncature à la cible.
    let mut parties: Vec<Vec<Ligne>> = Vec::new();
    let mut utilisables = 0usize;
    let mut compteur_graines = 0u64;
    while utilisables < opt.positions {
        let manque = opt.positions - utilisables;
        let n_parties =
            (manque / POSITIONS_PAR_PARTIE_ESTIMEES + 1).max(rayon::current_num_threads());
        let vague: Vec<Vec<Ligne>> = (0..n_parties)
            .into_par_iter()
            .with_max_len(1)
            .map(|i| {
                joue_partie(
                    &net,
                    &opt,
                    &opts_recherche,
                    &pool,
                    opt.seed.wrapping_add(compteur_graines + i as u64),
                )
            })
            .collect();
        compteur_graines += n_parties as u64;
        for p in vague {
            utilisables += p.iter().filter(|l| l.label1.is_some()).count();
            parties.push(p);
            if utilisables >= opt.positions {
                break;
            }
        }
        print!("\r[corpus] {} / {} positions etiquetees", utilisables.min(opt.positions), opt.positions);
        let _ = std::io::stdout().flush();
    }
    println!();
    // Troncature : parties entières puis queue de la dernière, pour que la
    // 2e passe rejoue exactement les requêtes conservées.
    {
        let mut cumul = 0usize;
        for p in parties.iter_mut() {
            if cumul >= opt.positions {
                p.clear();
                continue;
            }
            let mut garde = p.len();
            let mut c = cumul;
            for (k, l) in p.iter().enumerate() {
                if c >= opt.positions {
                    garde = k;
                    break;
                }
                if l.label1.is_some() {
                    c += 1;
                }
            }
            p.truncate(garde);
            cumul = c;
        }
        parties.retain(|p| !p.is_empty());
    }
    let n_parties_jouees = parties.len();
    let n_lignes: usize = parties.iter().map(|p| p.len()).sum();
    println!(
        "corpus : {n_parties_jouees} parties conservees, {n_lignes} positions (dont {} avec label1)",
        parties
            .iter()
            .flat_map(|p| p.iter())
            .filter(|l| l.label1.is_some())
            .count()
    );

    // 2. Deuxième étiquette, partie par partie (mêmes séquences de requêtes).
    let labels2: Vec<Vec<Option<f32>>> = parties
        .par_iter()
        .with_max_len(1)
        .map(|p| etiquette_partie_bis(p, &opt, &pool))
        .collect();
    println!("deuxieme etiquetage termine");

    // 3. Assemblage des triplets complets + CSV.
    let mut triplets: Vec<(f64, f64, f64)> = Vec::with_capacity(n_lignes);
    let mut ecartees = 0usize;
    for (p, l2s) in parties.iter().zip(&labels2) {
        for (l, l2) in p.iter().zip(l2s) {
            match (l.label1, l2) {
                (Some(a), Some(b)) => {
                    triplets.push((l.v_champion as f64, a as f64, *b as f64))
                }
                _ => ecartees += 1,
            }
        }
    }
    if triplets.is_empty() {
        eprintln!("aucun triplet complet : oracle muet ?");
        std::process::exit(1);
    }
    {
        // --append : ajoute au fichier existant (tranches), entête seulement
        // si le fichier est neuf ou vide.
        let existant = opt.append
            && std::fs::metadata(&opt.out).map(|m| m.len() > 0).unwrap_or(false);
        let mut fichier = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(opt.append)
            .truncate(!opt.append)
            .open(&opt.out)
            .unwrap_or_else(|e| {
                eprintln!("ouverture de {} : {e}", opt.out);
                std::process::exit(2);
            });
        if !existant {
            let _ = writeln!(fichier, "v_champion,label1,label2");
        }
        for (v, a, b) in &triplets {
            let _ = writeln!(fichier, "{v:.6},{a:.6},{b:.6}");
        }
    }
    println!(
        "csv : {} triplets -> {} ({ecartees} lignes ecartees, oracle muet)",
        triplets.len(),
        opt.out
    );

    // 4. Analyse.
    let v: Vec<f64> = triplets.iter().map(|t| t.0).collect();
    let l1: Vec<f64> = triplets.iter().map(|t| t.1).collect();
    let l2: Vec<f64> = triplets.iter().map(|t| t.2).collect();
    println!();
    println!(
        "correlations v_champion vs label1 : Pearson {:.4} | Spearman {:.4}",
        pearson(&v, &l1),
        spearman(&v, &l1)
    );
    let identiques = l1.iter().zip(&l2).filter(|(a, b)| a == b).count();
    println!(
        "labels : |l1 - l2| moyen {:.5} | paires identiques {:.1} % ({identiques} / {})",
        moyenne(&l1.iter().zip(&l2).map(|(a, b)| (a - b).abs()).collect::<Vec<_>>()),
        100.0 * identiques as f64 / l1.len() as f64,
        l1.len()
    );

    // Table de calibration : 20 quantiles de v_champion.
    println!();
    println!("table de calibration (20 quantiles de v_champion) :");
    println!("{:>4} {:>6} {:>12} {:>12} {:>12}", "q", "n", "v_moyen", "label_moyen", "ecart");
    let mut idx: Vec<usize> = (0..triplets.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).expect("valeurs finies"));
    let n = idx.len();
    for q in 0..20 {
        let deb = q * n / 20;
        let fin = (q + 1) * n / 20;
        if deb >= fin {
            continue;
        }
        let tranche = &idx[deb..fin];
        let vm = moyenne(&tranche.iter().map(|&i| v[i]).collect::<Vec<_>>());
        let lm = moyenne(&tranche.iter().map(|&i| l1[i]).collect::<Vec<_>>());
        println!(
            "{:>4} {:>6} {:>12.4} {:>12.4} {:>12.4}",
            q + 1,
            tranche.len(),
            vm,
            lm,
            lm - vm
        );
    }

    // Décomposition de la loss + garde-fou sur deux moitiés (indices pairs /
    // impairs : les deux moitiés couvrent les mêmes parties et la même phase
    // du corpus, seule la variance d'échantillonnage les sépare).
    println!();
    imprime_decomposition("decomposition de la loss", &decompose(&triplets));
    let pairs: Vec<(f64, f64, f64)> = triplets.iter().copied().step_by(2).collect();
    let impairs: Vec<(f64, f64, f64)> = triplets.iter().copied().skip(1).step_by(2).collect();
    println!();
    imprime_decomposition("garde-fou, moitie A (indices pairs)", &decompose(&pairs));
    println!();
    imprime_decomposition("garde-fou, moitie B (indices impairs)", &decompose(&impairs));

    // Mode --fit sans --from-csv : table ajustée sur les mesures du run.
    if !opt.fit.is_empty() {
        println!();
        ajuste_et_ecrit_table(&triplets, &opt.fit);
    }

    println!();
    println!("duree totale : {:.1} s", debut.elapsed().as_secs_f64());
}
