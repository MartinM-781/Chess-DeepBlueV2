//! Greffe « roi8 » : transplantation EXACTE d'un champion Classique773 dans le
//! schéma RoiZones8. Les 6149 features roi-zones englobent strictement les 773
//! classiques (mêmes plans pièce-case, dupliqués par zone du roi ; mêmes 5
//! scalaires en queue) : répliquer les 768 colonnes de plans du prof dans
//! chacun des 8 blocs de zone et recopier extras, biais et couches supérieures
//! verbatim donne un réseau roi8 fonctionnellement IDENTIQUE au prof sur toute
//! position — identité mathématique, PROUVÉE ici par une parité intégrée.
//!
//! Deux modes :
//!   greffe --teacher <773.bin> --out <roi8.bin>      transplantation + parité
//!   greffe --diagnose <roi8.bin> --teacher <773.bin> MSE net-vs-prof PAR ZONE
//!                                                    du roi du trait (autopsie
//!                                                    des « zones-poubelle »)
//! Options communes : --positions 50000 (taille du mélange), --seed 0.

use rand::rngs::StdRng;
use rand::SeedableRng;
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
            "--positions" => opt.positions = parse_valeur(&valeur(&args, i, &nom), &nom),
            "--seed" => opt.seed = parse_valeur(&valeur(&args, i, &nom), &nom),
            autre => {
                eprintln!("option inconnue : {autre}");
                std::process::exit(2);
            }
        }
        i += 2;
    }
    if opt.out.is_some() == opt.diagnose.is_some() {
        eprintln!("usage : greffe --teacher <773.bin> --out <roi8.bin>");
        eprintln!("        greffe --diagnose <roi8.bin> --teacher <773.bin>");
        eprintln!("(exactement UN des deux modes --out / --diagnose)");
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

fn main() {
    echec::pleine_puissance();
    let opt = parse_options();

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
