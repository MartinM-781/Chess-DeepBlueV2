//! Greffe « roi8 » : transplantation EXACTE d'un champion Classique773 dans le
//! schéma RoiZones8. Les 6149 features roi-zones englobent strictement les 773
//! classiques (mêmes plans pièce-case, dupliqués par zone du roi ; mêmes 5
//! scalaires en queue) : répliquer les 768 colonnes de plans du prof dans
//! chacun des 8 blocs de zone et recopier extras, biais et couches supérieures
//! verbatim donne un réseau roi8 fonctionnellement IDENTIQUE au prof sur toute
//! position — identité mathématique, PROUVÉE ici par une parité intégrée.
//!
//! Cinq modes :
//!   greffe --teacher <773.bin> --out <roi8.bin>      transplantation + parité
//!   greffe --diagnose <roi8.bin> --teacher <773.bin> MSE net-vs-prof PAR ZONE
//!                                                    du roi du trait (autopsie
//!                                                    des « zones-poubelle »)
//!                                                    + moyenne/écart-type des
//!                                                    sorties et pente de
//!                                                    régression net ~ prof
//!   greffe --compare <a.bin> <b.bin>                 RMS et max|a-b| par couche
//!                                                    (poids et biais séparés)
//!                                                    + RMS global pondéré
//!   greffe --bruit <sigma> --depuis <src.bin> --out <dst.bin>
//!                                                    clone de src, bruit
//!                                                    gaussien N(0, sigma²) sur
//!                                                    TOUS poids et biais
//!   greffe --echelle <c> --depuis <src.bin> --out <dst.bin>
//!                                                    clone de src, poids ET
//!                                                    biais de la DERNIÈRE
//!                                                    couche multipliés par c :
//!                                                    la sortie devient
//!                                                    tanh(c·z), monotone en la
//!                                                    sortie d'origine (ordres
//!                                                    rigoureusement préservés)
//! Options communes : --positions 50000 (taille du mélange), --seed 0.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use shakmaty::fen::Fen;
use shakmaty::{Chess, Color, EnPassantMode, Position};

use echec::bots::{Bot, RandomBot};
use echec::departs;
use echec::features::N_FEATURES;
use echec::features_roi::{zone_roi, N_FEATURES_ROI, N_ZONES_ROI};
use echec::nn::{evalue_position, Mlp, SchemaFeatures};

/// Volume par défaut du mélange de positions (parité et diagnostic).
const N_POSITIONS_DEFAUT: usize = 50_000;
/// Seuil d'ÉCHEC de la parité : au-delà, la greffe est fausse.
const SEUIL_PARITE: f32 = 1e-4;
/// Plis maximum d'une partie aléatoire du mélange.
const MAX_PLIS: usize = 120;

// ---------------------------------------------------------------------------
// Options (parse maison sur std::env::args, comme distill/train/serve)
// ---------------------------------------------------------------------------

struct Options {
    teacher: String,
    /// Mode greffe : destination du réseau roi8 (fichier NEUF exigé).
    out: Option<String>,
    /// Mode diagnostic : réseau roi8 à autopsier.
    diagnose: Option<String>,
    /// Mode comparaison : les deux réseaux à mesurer.
    compare: Option<(String, String)>,
    /// Mode bruit : écart-type du bruit gaussien.
    bruit: Option<f32>,
    /// Mode échelle : facteur multiplicatif de la dernière couche.
    echelle: Option<f32>,
    /// Modes bruit et échelle : réseau source à cloner.
    depuis: Option<String>,
    positions: usize,
    seed: u64,
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
        teacher: "models/chess_best.bin".to_string(),
        out: None,
        diagnose: None,
        compare: None,
        bruit: None,
        echelle: None,
        depuis: None,
        positions: N_POSITIONS_DEFAUT,
        seed: 0,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let nom = args[i].clone();
        match nom.as_str() {
            "--teacher" => opt.teacher = valeur(&args, i, &nom),
            "--out" => opt.out = Some(valeur(&args, i, &nom)),
            "--diagnose" => opt.diagnose = Some(valeur(&args, i, &nom)),
            "--compare" => {
                // Deux valeurs consécutives : les deux réseaux à comparer.
                let a = valeur(&args, i, &nom);
                let b = valeur(&args, i + 1, &nom);
                opt.compare = Some((a, b));
                i += 1;
            }
            "--bruit" => opt.bruit = Some(parse_valeur(&valeur(&args, i, &nom), &nom)),
            "--echelle" => opt.echelle = Some(parse_valeur(&valeur(&args, i, &nom), &nom)),
            "--depuis" => opt.depuis = Some(valeur(&args, i, &nom)),
            "--positions" => opt.positions = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            autre => {
                eprintln!("option inconnue : {autre}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    // Exactement UN mode : greffe (--out seul), diagnostic, comparaison,
    // bruit, échelle.
    let n_modes = [
        opt.out.is_some() && opt.bruit.is_none() && opt.echelle.is_none(),
        opt.diagnose.is_some(),
        opt.compare.is_some(),
        opt.bruit.is_some(),
        opt.echelle.is_some(),
    ]
    .iter()
    .filter(|&&m| m)
    .count();
    if n_modes != 1 {
        eprintln!("usage : greffe --teacher <773.bin> --out <roi8.bin>");
        eprintln!("        greffe --diagnose <roi8.bin> --teacher <773.bin>");
        eprintln!("        greffe --compare <a.bin> <b.bin>");
        eprintln!("        greffe --bruit <sigma> --depuis <src.bin> --out <dst.bin>");
        eprintln!("        greffe --echelle <c> --depuis <src.bin> --out <dst.bin>");
        eprintln!("(exactement UN mode)");
        std::process::exit(2);
    }
    if let Some(sigma) = opt.bruit {
        if !(sigma > 0.0) || !sigma.is_finite() {
            eprintln!("option --bruit : sigma doit etre fini et > 0 (recu {sigma})");
            std::process::exit(2);
        }
        if opt.depuis.is_none() || opt.out.is_none() {
            eprintln!("mode --bruit : --depuis <src.bin> et --out <dst.bin> obligatoires");
            std::process::exit(2);
        }
    } else if let Some(c) = opt.echelle {
        if !(c > 0.0) || !c.is_finite() {
            eprintln!("option --echelle : c doit etre fini et > 0 (recu {c})");
            std::process::exit(2);
        }
        if opt.depuis.is_none() || opt.out.is_none() {
            eprintln!("mode --echelle : --depuis <src.bin> et --out <dst.bin> obligatoires");
            std::process::exit(2);
        }
    } else if opt.depuis.is_some() {
        eprintln!("option --depuis : reservee aux modes --bruit et --echelle");
        std::process::exit(2);
    }
    if opt.positions == 0 {
        eprintln!("option --positions : 0 refuse (il faut des positions a comparer)");
        std::process::exit(2);
    }
    opt
}

// ---------------------------------------------------------------------------
// Transplantation
// ---------------------------------------------------------------------------

/// Correspondance d'indices, DÉRIVÉE des deux encodeurs :
/// - `features::encode` : pièce → `plan*64 + case_vue` (plans 0..12, ordre
///   P,N,B,R,Q,K nôtres puis leurs), extras 768..773 (notre O-O, notre O-O-O,
///   leur O-O, leur O-O-O, en passant légale) ;
/// - `features_roi::actifs_perspective` : pièce → `zone*768 + plan*64 +
///   case_vue` (MÊMES plans, MÊME miroir case^56, MÊME bascule de couleurs),
///   extras 6144..6149 (MÊME ordre).
/// Une position n'activant qu'UNE zone, la pré-activation creuse
/// biais + Σ colonnes actives est identique à celle du prof dès lors que
/// chaque bloc de zone contient la copie des 768 colonnes de plans du prof et
/// que les 5 colonnes d'extras sont recopiées en queue. Biais et couches
/// supérieures verbatim : réseau fonctionnellement identique.
fn greffer(prof: &Mlp) -> Mlp {
    assert_eq!(
        prof.schema(),
        SchemaFeatures::Classique773,
        "greffe : le prof doit etre au schema Classique773"
    );
    let n1 = prof.sizes[1];
    let mut sizes = prof.sizes.clone();
    sizes[0] = N_FEATURES_ROI;

    // Couche d'entrée : réplication des 768 colonnes de plans dans les 8
    // blocs de zone, extras recopiées en queue (768+k → 6144+k).
    let w_prof = &prof.weights[0];
    let mut w0 = vec![0.0f32; n1 * N_FEATURES_ROI];
    for j in 0..n1 {
        let src = &w_prof[j * N_FEATURES..(j + 1) * N_FEATURES];
        let dst = &mut w0[j * N_FEATURES_ROI..(j + 1) * N_FEATURES_ROI];
        for zone in 0..N_ZONES_ROI {
            dst[zone * 768..(zone + 1) * 768].copy_from_slice(&src[..768]);
        }
        dst[N_ZONES_ROI * 768..].copy_from_slice(&src[768..N_FEATURES]);
    }

    // Couches supérieures et TOUS les biais : verbatim.
    let mut weights = vec![w0];
    weights.extend(prof.weights[1..].iter().cloned());
    let biases = prof.biases.clone();

    // Moments Adam à zéro, steps 0, pas_colonnes à zéros : réseau « neuf »
    // du point de vue de l'optimiseur.
    let zeros_w: Vec<Vec<f32>> = weights.iter().map(|w| vec![0.0; w.len()]).collect();
    let zeros_b: Vec<Vec<f32>> = biases.iter().map(|b| vec![0.0; b.len()]).collect();
    Mlp {
        sizes,
        weights,
        biases,
        adam_mw: zeros_w.clone(),
        adam_vw: zeros_w,
        adam_mb: zeros_b.clone(),
        adam_vb: zeros_b,
        steps: 0,
        pas_colonnes: vec![0u64; N_FEATURES_ROI],
    }
}

// ---------------------------------------------------------------------------
// Mélange de positions (parité et diagnostic)
// ---------------------------------------------------------------------------

/// Positions d'UNE partie de coups aléatoires (RandomBot) depuis un départ
/// choisi par la graine : 3 fois sur 5 la position initiale (les rois y
/// vagabondent vite sous coups aléatoires : toutes les zones sont visitées),
/// 1 fois sur 5 une ouverture du livre, 1 fois sur 5 une finale générée
/// (`src/departs.rs`). Le départ lui-même est inclus.
fn partie_positions(graine: u64) -> Vec<Chess> {
    let mut rng = StdRng::seed_from_u64(graine.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let depart = match graine % 5 {
        3 => departs::tirage(&mut rng, 1.0, 0.0).pos, // ouverture du livre
        4 => departs::tirage(&mut rng, 0.0, 1.0).pos, // finale générée
        _ => Chess::default(),
    };
    let mut bot = RandomBot::new(graine.wrapping_add(0xA5A5_A5A5));
    let mut pos = depart;
    let mut sortie = Vec::with_capacity(MAX_PLIS + 1);
    sortie.push(pos.clone());
    for _ in 0..MAX_PLIS {
        if pos.legal_moves().is_empty()
            || pos.is_insufficient_material()
            || pos.halfmoves() >= 100
        {
            break;
        }
        let coup = bot.choose(&pos).expect("coups légaux non vides");
        pos = pos.play(&coup).expect("coup légal");
        sortie.push(pos.clone());
    }
    sortie
}

/// Exactement `n` positions variées, par vagues de parties parallèles (rayon,
/// une partie par tâche), graines déterministes `graine_base + compteur`.
fn positions_melange(n: usize, graine_base: u64) -> Vec<Chess> {
    let mut positions = Vec::with_capacity(n);
    let mut compteur = 0u64;
    while positions.len() < n {
        let manque = n - positions.len();
        // ~40 positions par partie en pratique ; au moins une vague pleine.
        let n_parties = (manque / 40 + 1).max(rayon::current_num_threads());
        let parties: Vec<Vec<Chess>> = (0..n_parties)
            .into_par_iter()
            .with_max_len(1)
            .map(|i| partie_positions(graine_base.wrapping_add(compteur + i as u64)))
            .collect();
        compteur += n_parties as u64;
        for p in parties {
            let prendre = (n - positions.len()).min(p.len());
            positions.extend(p.into_iter().take(prendre));
            if positions.len() >= n {
                break;
            }
        }
    }
    positions
}

/// Évaluations parallèles des deux réseaux sur le mélange :
/// (sorties du prof, sorties de l'autre), dans l'ordre des positions.
fn evaluations(prof: &Mlp, autre: &Mlp, positions: &[Chess]) -> (Vec<f32>, Vec<f32>) {
    let y_prof: Vec<f32> = positions
        .par_iter()
        .map_init(Vec::new, |tampon, pos| evalue_position(prof, pos, tampon))
        .collect();
    let y_autre: Vec<f32> = positions
        .par_iter()
        .map_init(Vec::new, |tampon, pos| evalue_position(autre, pos, tampon))
        .collect();
    (y_prof, y_autre)
}

/// Zone (0..8) du roi du camp AU TRAIT, vue dans sa perspective (miroir
/// case^56 si les noirs jouent) — la zone qui choisit le bloc de features de
/// `features_roi::actifs` pour cette position.
fn zone_du_trait(pos: &Chess) -> usize {
    let roi = pos
        .board()
        .king_of(pos.turn())
        .expect("position légale : chaque camp a un roi");
    let case = if pos.turn() == Color::Black {
        usize::from(roi) ^ 56
    } else {
        usize::from(roi)
    };
    zone_roi(case)
}

// ---------------------------------------------------------------------------
// Les deux modes
// ---------------------------------------------------------------------------

/// Greffe + parité intégrée. Renvoie le code de sortie du processus.
fn mode_greffe(opt: &Options, prof: &Mlp, out: &str) -> i32 {
    // Jamais écraser un modèle existant : --out doit être un fichier NEUF.
    if std::path::Path::new(out).exists() {
        eprintln!("--out {out} : le fichier existe deja, greffe refusee (jamais d'ecrasement)");
        return 2;
    }

    let greffe = greffer(prof);
    println!(
        "greffe : prof {:?} ({:?}) -> {:?} ({:?})",
        prof.sizes,
        prof.schema(),
        greffe.sizes,
        greffe.schema()
    );
    if let Err(e) = greffe.save(out) {
        eprintln!("sauvegarde de {out} : {e}");
        return 2;
    }
    println!("reseau greffe sauvegarde -> {out}");

    // Parité intégrée : le greffé doit être le prof, à l'ordre des sommations
    // flottantes près (dense 773 colonnes vs somme creuse des actives).
    println!("parite : generation de {} positions variees...", opt.positions);
    let positions = positions_melange(opt.positions, opt.seed);
    let (y_prof, y_greffe) = evaluations(prof, &greffe, &positions);

    let mut max_diff = 0.0f32;
    let mut i_max = 0usize;
    let mut somme = 0.0f64;
    let mut fautives: Vec<usize> = Vec::new(); // indices à |diff| > seuil
    for (i, (a, b)) in y_prof.iter().zip(&y_greffe).enumerate() {
        let d = (a - b).abs();
        somme += d as f64;
        if d > max_diff {
            max_diff = d;
            i_max = i;
        }
        if d > SEUIL_PARITE {
            fautives.push(i);
        }
    }
    let moyenne = somme / positions.len() as f64;
    println!(
        "parite sur {} positions : max|diff| = {max_diff:.3e} | moyenne|diff| = {moyenne:.3e}",
        positions.len()
    );
    println!(
        "position du max : {}",
        Fen::from_position(positions[i_max].clone(), EnPassantMode::Legal)
    );

    if max_diff > SEUIL_PARITE {
        eprintln!(
            "ECHEC de la parite : max|diff| {max_diff:.3e} > {SEUIL_PARITE:.0e} \
             ({} positions fautives), exemples :",
            fautives.len()
        );
        for &i in fautives.iter().take(10) {
            eprintln!(
                "  |diff| = {:.3e}  prof = {:+.6}  greffe = {:+.6}  {}",
                (y_prof[i] - y_greffe[i]).abs(),
                y_prof[i],
                y_greffe[i],
                Fen::from_position(positions[i].clone(), EnPassantMode::Legal)
            );
        }
        1
    } else {
        println!("parite OK (seuil {SEUIL_PARITE:.0e}) : la greffe est le prof, au flottant pres");
        0
    }
}

/// Autopsie par zone : MSE net-vs-prof ventilé par zone du roi du trait.
fn mode_diagnose(opt: &Options, prof: &Mlp, chemin_net: &str) -> i32 {
    let net = match Mlp::load(chemin_net) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("chargement de {chemin_net} : {e}");
            return 2;
        }
    };
    if net.schema() != SchemaFeatures::RoiZones8 {
        eprintln!(
            "--diagnose {chemin_net} : schema {:?}, attendu RoiZones8",
            net.schema()
        );
        return 2;
    }
    println!(
        "diagnostic : net {chemin_net} {:?} vs prof {} {:?} | {} positions",
        net.sizes, opt.teacher, prof.sizes, opt.positions
    );

    let positions = positions_melange(opt.positions, opt.seed);
    let (y_prof, y_net) = evaluations(prof, &net, &positions);

    // Statistiques d'échelle : moyenne et écart-type des sorties des deux
    // réseaux, puis pente de régression linéaire y_net ~ y_prof (la « dérive
    // d'échelle » mesurée : cov(prof, net) / var(prof)).
    let (m_prof, s_prof) = moyenne_ecart_type(&y_prof);
    let (m_net, s_net) = moyenne_ecart_type(&y_net);
    let mut cov = 0.0f64;
    let mut var = 0.0f64;
    for (p, n) in y_prof.iter().zip(&y_net) {
        let dp = *p as f64 - m_prof;
        cov += dp * (*n as f64 - m_net);
        var += dp * dp;
    }
    let pente = cov / var;
    println!("sorties prof : moyenne = {m_prof:+.6} | ecart-type = {s_prof:.6}");
    println!("sorties net  : moyenne = {m_net:+.6} | ecart-type = {s_net:.6}");
    println!("pente de regression net ~ prof (derive d'echelle) : {pente:.6}");

    // Ventilation par zone du roi du trait : n, somme des carrés, gros écarts.
    let mut n_zone = [0usize; N_ZONES_ROI];
    let mut somme_carres = [0.0f64; N_ZONES_ROI];
    let mut gros_ecarts = [0usize; N_ZONES_ROI];
    for (i, pos) in positions.iter().enumerate() {
        let z = zone_du_trait(pos);
        let e = (y_net[i] - y_prof[i]) as f64;
        n_zone[z] += 1;
        somme_carres[z] += e * e;
        if e.abs() > 0.1 {
            gros_ecarts[z] += 1;
        }
    }

    // Sortie triée par MSE décroissant : les zones-poubelle en tête.
    let mut lignes: Vec<(usize, usize, f64, f64)> = (0..N_ZONES_ROI)
        .map(|z| {
            let mse = if n_zone[z] > 0 { somme_carres[z] / n_zone[z] as f64 } else { 0.0 };
            let pct = if n_zone[z] > 0 {
                100.0 * gros_ecarts[z] as f64 / n_zone[z] as f64
            } else {
                0.0
            };
            (z, n_zone[z], mse, pct)
        })
        .collect();
    lignes.sort_by(|a, b| b.2.partial_cmp(&a.2).expect("MSE finis"));

    println!("zone | n positions |      MSE | % |ecart| > 0.1");
    for (z, n, mse, pct) in &lignes {
        println!("   {z} | {n:>11} | {mse:.6} | {pct:>6.2} %");
    }
    let mse_global: f64 = somme_carres.iter().sum::<f64>() / positions.len() as f64;
    println!("global : MSE {mse_global:.6} sur {} positions", positions.len());
    0
}

/// Moyenne et écart-type (population) d'un vecteur de sorties.
fn moyenne_ecart_type(y: &[f32]) -> (f64, f64) {
    let n = y.len() as f64;
    let moyenne = y.iter().map(|&v| v as f64).sum::<f64>() / n;
    let variance = y.iter().map(|&v| (v as f64 - moyenne).powi(2)).sum::<f64>() / n;
    (moyenne, variance.sqrt())
}

/// RMS et max|d| d'un vecteur de différences a-b.
fn rms_max(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut somme_carres = 0.0f64;
    let mut max = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        somme_carres += d * d;
        max = max.max(d.abs());
    }
    (somme_carres / a.len() as f64, max) // somme/n : la racine est prise par l'appelant
}

/// Comparaison paramètre à paramètre de deux réseaux : RMS(a-b) et max|a-b|
/// par couche (poids et biais séparés), puis RMS global pondéré par le nombre
/// de paramètres. L'outil de mesure du déplacement infligé par l'entraînement.
fn mode_compare(chemin_a: &str, chemin_b: &str) -> i32 {
    let a = match Mlp::load(chemin_a) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("chargement de {chemin_a} : {e}");
            return 2;
        }
    };
    let b = match Mlp::load(chemin_b) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("chargement de {chemin_b} : {e}");
            return 2;
        }
    };
    if a.sizes != b.sizes {
        eprintln!(
            "--compare : architectures incompatibles ({:?} vs {:?}), comparaison refusee",
            a.sizes, b.sizes
        );
        return 2;
    }
    println!("compare : {chemin_a} vs {chemin_b} | architecture {:?}", a.sizes);

    let mut somme_carres_glob = 0.0f64;
    let mut n_glob = 0usize;
    let mut max_glob = 0.0f64;
    for l in 0..a.weights.len() {
        let (mc_w, max_w) = rms_max(&a.weights[l], &b.weights[l]);
        let (mc_b, max_b) = rms_max(&a.biases[l], &b.biases[l]);
        println!(
            "couche {l} poids ({:>9} params) : RMS = {:.6e} | max|diff| = {:.6e}",
            a.weights[l].len(),
            mc_w.sqrt(),
            max_w
        );
        println!(
            "couche {l} biais ({:>9} params) : RMS = {:.6e} | max|diff| = {:.6e}",
            a.biases[l].len(),
            mc_b.sqrt(),
            max_b
        );
        somme_carres_glob += mc_w * a.weights[l].len() as f64 + mc_b * a.biases[l].len() as f64;
        n_glob += a.weights[l].len() + a.biases[l].len();
        max_glob = max_glob.max(max_w).max(max_b);
    }
    println!(
        "global ({n_glob} params) : RMS pondere = {:.6e} | max|diff| = {max_glob:.6e}",
        (somme_carres_glob / n_glob as f64).sqrt()
    );
    0
}

/// Tirage gaussien N(0, 1) par Box-Muller (rand 0.8 n'embarque pas de normale).
fn gaussien(rng: &mut StdRng) -> f64 {
    let u1: f64 = 1.0 - rng.gen::<f64>(); // dans (0, 1] : ln(u1) est fini
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Clone de src avec bruit gaussien N(0, sigma²) ajouté à TOUS les poids et
/// biais — zéro entraînement, zéro direction : le perturbateur pur de la
/// théorie de l'aiguille. Moments Adam, steps et pas_colonnes copiés tels
/// quels. Refuse d'écraser un fichier existant.
fn mode_bruit(sigma: f32, chemin_src: &str, chemin_dst: &str, graine: u64) -> i32 {
    if std::path::Path::new(chemin_dst).exists() {
        eprintln!("--out {chemin_dst} : le fichier existe deja, bruit refuse (jamais d'ecrasement)");
        return 2;
    }
    let mut net = match Mlp::load(chemin_src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("chargement de {chemin_src} : {e}");
            return 2;
        }
    };

    let mut rng = StdRng::seed_from_u64(graine);
    let mut n_perturbes = 0usize;
    for w in net.weights.iter_mut().chain(net.biases.iter_mut()) {
        for x in w.iter_mut() {
            *x += (sigma as f64 * gaussien(&mut rng)) as f32;
            n_perturbes += 1;
        }
    }

    if let Err(e) = net.save(chemin_dst) {
        eprintln!("sauvegarde de {chemin_dst} : {e}");
        return 2;
    }
    println!(
        "bruit : {chemin_src} + N(0, {sigma}^2) sur {n_perturbes} parametres \
         (graine {graine}) -> {chemin_dst}"
    );
    0
}

/// Clone de src dont la DERNIÈRE couche (poids ET biais) est multipliée par c :
/// la pré-activation de sortie devient c·z, la sortie tanh(c·z) — strictement
/// monotone en la sortie d'origine, l'ordre des évaluations est rigoureusement
/// préservé. Seule l'ÉCHELLE change : le testeur de la théorie des marges
/// fixes de la recherche. Moments Adam, steps et pas_colonnes copiés tels
/// quels. Refuse d'écraser un fichier existant.
fn mode_echelle(c: f32, chemin_src: &str, chemin_dst: &str) -> i32 {
    if std::path::Path::new(chemin_dst).exists() {
        eprintln!(
            "--out {chemin_dst} : le fichier existe deja, echelle refusee (jamais d'ecrasement)"
        );
        return 2;
    }
    let mut net = match Mlp::load(chemin_src) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("chargement de {chemin_src} : {e}");
            return 2;
        }
    };

    let w_dernier = net.weights.last_mut().expect("reseau non vide");
    for x in w_dernier.iter_mut() {
        *x *= c;
    }
    let b_dernier = net.biases.last_mut().expect("reseau non vide");
    for x in b_dernier.iter_mut() {
        *x *= c;
    }
    let n_modifies = w_dernier.len() + b_dernier.len();

    if let Err(e) = net.save(chemin_dst) {
        eprintln!("sauvegarde de {chemin_dst} : {e}");
        return 2;
    }
    println!(
        "echelle : {chemin_src} x {c} sur la derniere couche ({n_modifies} parametres) \
         -> {chemin_dst}"
    );
    0
}

fn main() {
    echec::pleine_puissance();
    let opt = parse_options();

    // Modes autonomes (sans prof) : comparaison et bruit.
    if let Some((a, b)) = &opt.compare {
        std::process::exit(mode_compare(a, b));
    }
    if let Some(sigma) = opt.bruit {
        let src = opt.depuis.as_ref().expect("mode bruit : --depuis verifie au parse");
        let dst = opt.out.as_ref().expect("mode bruit : --out verifie au parse");
        std::process::exit(mode_bruit(sigma, src, dst, opt.seed));
    }
    if let Some(c) = opt.echelle {
        let src = opt.depuis.as_ref().expect("mode echelle : --depuis verifie au parse");
        let dst = opt.out.as_ref().expect("mode echelle : --out verifie au parse");
        std::process::exit(mode_echelle(c, src, dst));
    }

    let prof = Mlp::load(&opt.teacher).unwrap_or_else(|e| {
        eprintln!("chargement du prof {} : {e}", opt.teacher);
        std::process::exit(2);
    });
    if prof.schema() != SchemaFeatures::Classique773 {
        eprintln!(
            "--teacher {} : schema {:?}, attendu Classique773",
            opt.teacher,
            prof.schema()
        );
        std::process::exit(2);
    }

    let code = if let Some(out) = &opt.out {
        mode_greffe(&opt, &prof, out)
    } else {
        mode_diagnose(&opt, &prof, opt.diagnose.as_ref().expect("mode diagnose"))
    };
    std::process::exit(code);
}
