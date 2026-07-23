//! Réseau de valeur : MLP maison 773 → 512 → 64 → 1 (ReLU cachées, tanh en sortie),
//! optimiseur Adam, sérialisation binaire maison (pas de dépendance).
//! `forward_*` doit être utilisable depuis plusieurs threads (&self, aucune mutation).

use std::io::{ErrorKind, Read, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rayon::prelude::*;

use crate::features::N_FEATURES;

/// Hyperparamètres Adam (fixes, seul le taux d'apprentissage est passé en argument).
const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Magic des fichiers modèle (8 octets), tout le reste est en little-endian.
const MAGIC: &[u8; 8] = b"ECHECNN1";

pub struct Mlp {
    /// Tailles des couches, ex. [773, 512, 64, 1].
    pub sizes: Vec<usize>,
    /// Poids par couche (row-major : sortie × entrée), biais par couche.
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<Vec<f32>>,
    /// Moments Adam (m et v pour poids et biais, mêmes formes que weights/biases).
    pub adam_mw: Vec<Vec<f32>>,
    pub adam_vw: Vec<Vec<f32>>,
    pub adam_mb: Vec<Vec<f32>>,
    pub adam_vb: Vec<Vec<f32>>,
    /// Nombre de pas d'optimisation effectués (pour la correction de biais Adam).
    pub steps: u64,
}

/// Tirage gaussien standard par Box-Muller (rand 0.8 n'embarque pas de Normal).
fn gaussienne(rng: &mut StdRng) -> f32 {
    // u1 ∈ (0,1] pour éviter ln(0), u2 ∈ [0,1).
    let u1: f32 = 1.0 - rng.gen::<f32>();
    let u2: f32 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

impl Mlp {
    /// Réseau [N_FEATURES, 512, 64, 1], init He, graine déterministe.
    pub fn new(seed: u64) -> Self {
        Self::avec_tailles(vec![N_FEATURES, 512, 64, 1], seed)
    }

    /// Construction générique (privée : les tailles publiques sont figées par `new`,
    /// mais les tests utilisent de petits réseaux pour aller vite).
    fn avec_tailles(sizes: Vec<usize>, seed: u64) -> Self {
        assert!(sizes.len() >= 2, "il faut au moins une couche entrée→sortie");
        let mut rng = StdRng::seed_from_u64(seed);
        let n_couches = sizes.len() - 1;

        let mut weights = Vec::with_capacity(n_couches);
        let mut biases = Vec::with_capacity(n_couches);
        for l in 0..n_couches {
            let (n_in, n_out) = (sizes[l], sizes[l + 1]);
            // Init He : N(0, sqrt(2 / fan_in)), biais à zéro.
            let ecart = (2.0 / n_in as f32).sqrt();
            let w: Vec<f32> = (0..n_in * n_out)
                .map(|_| gaussienne(&mut rng) * ecart)
                .collect();
            weights.push(w);
            biases.push(vec![0.0; n_out]);
        }

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
        }
    }

    /// Passe avant d'une seule entrée en réutilisant deux tampons fournis.
    /// C'est LE chemin de calcul commun : forward_one et forward_batch l'appellent
    /// tous deux, garantissant des résultats bit-à-bit identiques.
    fn avancer(&self, x: &[f32], courant: &mut Vec<f32>, suivant: &mut Vec<f32>) -> f32 {
        debug_assert_eq!(x.len(), self.sizes[0]);
        courant.clear();
        courant.extend_from_slice(x);
        let n_couches = self.sizes.len() - 1;
        for l in 0..n_couches {
            let (n_in, n_out) = (self.sizes[l], self.sizes[l + 1]);
            let w = &self.weights[l];
            let b = &self.biases[l];
            suivant.clear();
            suivant.resize(n_out, 0.0);
            for j in 0..n_out {
                let ligne = &w[j * n_in..(j + 1) * n_in];
                let mut s = b[j];
                for k in 0..n_in {
                    s += ligne[k] * courant[k];
                }
                // ReLU sur les couches cachées, tanh en sortie.
                suivant[j] = if l + 1 == n_couches { s.tanh() } else { s.max(0.0) };
            }
            std::mem::swap(courant, suivant);
        }
        courant[0]
    }

    /// Capacité maximale nécessaire pour les tampons d'activation.
    fn taille_tampon(&self) -> usize {
        self.sizes.iter().copied().max().unwrap_or(0)
    }

    /// Valeur (dans [-1,1], perspective du trait) d'une position encodée.
    pub fn forward_one(&self, x: &[f32]) -> f32 {
        let cap = self.taille_tampon();
        let mut a = Vec::with_capacity(cap);
        let mut b = Vec::with_capacity(cap);
        self.avancer(x, &mut a, &mut b)
    }

    /// Passe avant sur un lot : `xs` contient n vecteurs concaténés (n × N_FEATURES).
    /// Les deux tampons d'activation sont alloués une seule fois puis réutilisés.
    pub fn forward_batch(&self, xs: &[f32], n: usize) -> Vec<f32> {
        let n_in = self.sizes[0];
        assert_eq!(xs.len(), n * n_in, "forward_batch: lot de mauvaise taille");
        let cap = self.taille_tampon();
        let mut a = Vec::with_capacity(cap);
        let mut b = Vec::with_capacity(cap);
        let mut sorties = Vec::with_capacity(n);
        for i in 0..n {
            sorties.push(self.avancer(&xs[i * n_in..(i + 1) * n_in], &mut a, &mut b));
        }
        sorties
    }

    /// Un pas d'Adam sur le lot (MSE entre tanh de sortie et `targets`), renvoie la loss.
    pub fn train_batch(&mut self, xs: &[f32], targets: &[f32], lr: f32) -> f32 {
        let n = targets.len();
        let n_in = self.sizes[0];
        assert!(n > 0, "train_batch: lot vide");
        assert_eq!(xs.len(), n * n_in, "train_batch: lot de mauvaise taille");
        let n_couches = self.sizes.len() - 1;

        // --- Passe avant, en conservant les activations de chaque couche. ---
        // acts[l] : activations (post non-linéarité) de la couche l+1, n × sizes[l+1].
        let mut acts: Vec<Vec<f32>> = Vec::with_capacity(n_couches);
        for l in 0..n_couches {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let derniere = l + 1 == n_couches;
            let prec: &[f32] = if l == 0 { xs } else { &acts[l - 1] };
            let w = &self.weights[l];
            let b = &self.biases[l];
            // Parallèle par échantillon (rayon) : chaque ligne de sortie est
            // indépendante. Depuis le NNUE, cette étape était devenue LE goulot
            // séquentiel des cycles (~30 s mono-thread par cycle).
            let mut a = vec![0.0f32; n * no];
            a.par_chunks_mut(no)
                .zip(prec.par_chunks(ni))
                .for_each(|(sortie, x)| {
                    for j in 0..no {
                        let ligne = &w[j * ni..(j + 1) * ni];
                        let mut s = b[j];
                        for k in 0..ni {
                            s += ligne[k] * x[k];
                        }
                        sortie[j] = if derniere { s.tanh() } else { s.max(0.0) };
                    }
                });
            acts.push(a);
        }

        // --- Loss MSE et delta de sortie (dL/dz, en traversant la tanh). ---
        let sorties = &acts[n_couches - 1]; // n × 1
        let mut loss = 0.0f32;
        let mut delta: Vec<f32> = Vec::with_capacity(n);
        for i in 0..n {
            let y = sorties[i];
            let ecart = y - targets[i];
            loss += ecart * ecart;
            // dL/dy = 2(y-t)/n ; dy/dz = 1 - tanh² = 1 - y².
            delta.push(2.0 * ecart / n as f32 * (1.0 - y * y));
        }
        loss /= n as f32;

        // --- Rétropropagation + mise à jour Adam couche par couche. ---
        self.steps += 1;
        // Corrections de biais calculées en f64 pour rester précises à grand `steps`.
        let c1 = (1.0 - (ADAM_B1 as f64).powi(self.steps as i32)) as f32;
        let c2 = (1.0 - (ADAM_B2 as f64).powi(self.steps as i32)) as f32;

        for l in (0..n_couches).rev() {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let prec: &[f32] = if l == 0 { xs } else { &acts[l - 1] };

            // Gradients des poids et biais de la couche l : accumulation
            // parallèle par échantillon avec réduction (un tampon de gradients
            // par thread, sommés à la fin — l'ordre de sommation flottante
            // change, sans conséquence : l'apprentissage est stochastique).
            let (grad_w, grad_b) = (0..n)
                .into_par_iter()
                .fold(
                    || (vec![0.0f32; no * ni], vec![0.0f32; no]),
                    |(mut gw_acc, mut gb_acc), i| {
                        let d = &delta[i * no..(i + 1) * no];
                        let x = &prec[i * ni..(i + 1) * ni];
                        for j in 0..no {
                            let dj = d[j];
                            if dj == 0.0 {
                                continue; // neurone ReLU éteint : rien à propager
                            }
                            gb_acc[j] += dj;
                            let gw = &mut gw_acc[j * ni..(j + 1) * ni];
                            for k in 0..ni {
                                gw[k] += dj * x[k];
                            }
                        }
                        (gw_acc, gb_acc)
                    },
                )
                .reduce(
                    || (vec![0.0f32; no * ni], vec![0.0f32; no]),
                    |(mut aw, mut ab), (bw, bb)| {
                        for (u, v) in aw.iter_mut().zip(&bw) {
                            *u += *v;
                        }
                        for (u, v) in ab.iter_mut().zip(&bb) {
                            *u += *v;
                        }
                        (aw, ab)
                    },
                );

            // Delta de la couche précédente (avec les poids AVANT mise à jour),
            // en traversant la ReLU : dérivée = 1 si activation > 0, sinon 0.
            if l > 0 {
                let w = &self.weights[l];
                let mut delta_prec = vec![0.0f32; n * ni];
                let delta_ref = &delta;
                delta_prec
                    .par_chunks_mut(ni)
                    .enumerate()
                    .for_each(|(i, dp)| {
                        let d = &delta_ref[i * no..(i + 1) * no];
                        for j in 0..no {
                            let dj = d[j];
                            if dj == 0.0 {
                                continue;
                            }
                            let ligne = &w[j * ni..(j + 1) * ni];
                            for k in 0..ni {
                                dp[k] += dj * ligne[k];
                            }
                        }
                        let a = &prec[i * ni..(i + 1) * ni];
                        for k in 0..ni {
                            if a[k] <= 0.0 {
                                dp[k] = 0.0;
                            }
                        }
                    });
                // Mise à jour Adam de la couche l (après le calcul de delta_prec).
                adam_maj(&mut self.weights[l], &mut self.adam_mw[l], &mut self.adam_vw[l],
                         &grad_w, lr, c1, c2);
                adam_maj(&mut self.biases[l], &mut self.adam_mb[l], &mut self.adam_vb[l],
                         &grad_b, lr, c1, c2);
                delta = delta_prec;
            } else {
                adam_maj(&mut self.weights[l], &mut self.adam_mw[l], &mut self.adam_vw[l],
                         &grad_w, lr, c1, c2);
                adam_maj(&mut self.biases[l], &mut self.adam_mb[l], &mut self.adam_vb[l],
                         &grad_b, lr, c1, c2);
            }
        }

        loss
    }

    /// Sérialisation binaire : magic "ECHECNN1", sizes, steps, poids, biais, moments.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(MAGIC)?;
        w.write_all(&(self.sizes.len() as u32).to_le_bytes())?;
        for &s in &self.sizes {
            w.write_all(&(s as u32).to_le_bytes())?;
        }
        w.write_all(&self.steps.to_le_bytes())?;
        // Ordre : tous les poids (couche par couche), tous les biais, puis les
        // moments dans l'ordre mw, vw, mb, vb.
        for groupe in [&self.weights, &self.biases, &self.adam_mw,
                       &self.adam_vw, &self.adam_mb, &self.adam_vb] {
            for couche in groupe {
                ecrire_f32s(&mut w, couche)?;
            }
        }
        w.flush()
    }

    pub fn load(path: &str) -> std::io::Result<Mlp> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "mauvais magic : pas un fichier ECHECNN1",
            ));
        }

        let n_sizes = lire_u32(&mut r)? as usize;
        if n_sizes < 2 {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "fichier modèle invalide : moins de deux couches",
            ));
        }
        let mut sizes = Vec::with_capacity(n_sizes);
        for _ in 0..n_sizes {
            sizes.push(lire_u32(&mut r)? as usize);
        }

        let mut steps_octets = [0u8; 8];
        r.read_exact(&mut steps_octets)?;
        let steps = u64::from_le_bytes(steps_octets);

        let n_couches = n_sizes - 1;
        let tailles_w: Vec<usize> = (0..n_couches).map(|l| sizes[l] * sizes[l + 1]).collect();
        let tailles_b: Vec<usize> = (0..n_couches).map(|l| sizes[l + 1]).collect();

        // Même ordre qu'à l'écriture : poids, biais, puis mw, vw, mb, vb.
        let lire_groupe = |r: &mut dyn Read, tailles: &[usize]| -> std::io::Result<Vec<Vec<f32>>> {
            tailles.iter().map(|&t| lire_f32s(r, t)).collect()
        };
        let weights = lire_groupe(&mut r, &tailles_w)?;
        let biases = lire_groupe(&mut r, &tailles_b)?;
        let adam_mw = lire_groupe(&mut r, &tailles_w)?;
        let adam_vw = lire_groupe(&mut r, &tailles_w)?;
        let adam_mb = lire_groupe(&mut r, &tailles_b)?;
        let adam_vb = lire_groupe(&mut r, &tailles_b)?;

        Ok(Mlp {
            sizes,
            weights,
            biases,
            adam_mw,
            adam_vw,
            adam_mb,
            adam_vb,
            steps,
        })
    }
}

/// Mise à jour Adam en place d'un tenseur de paramètres.
/// `c1`/`c2` sont les facteurs de correction de biais (1 - β^t) déjà calculés.
fn adam_maj(params: &mut [f32], m: &mut [f32], v: &mut [f32], grad: &[f32],
            lr: f32, c1: f32, c2: f32) {
    debug_assert!(params.len() == m.len() && m.len() == v.len() && v.len() == grad.len());
    for i in 0..params.len() {
        let g = grad[i];
        m[i] = ADAM_B1 * m[i] + (1.0 - ADAM_B1) * g;
        v[i] = ADAM_B2 * v[i] + (1.0 - ADAM_B2) * g * g;
        let m_chapeau = m[i] / c1;
        let v_chapeau = v[i] / c2;
        params[i] -= lr * m_chapeau / (v_chapeau.sqrt() + ADAM_EPS);
    }
}

/// Écrit un tableau de f32 en little-endian d'un seul bloc.
fn ecrire_f32s(w: &mut impl Write, valeurs: &[f32]) -> std::io::Result<()> {
    let mut octets = Vec::with_capacity(valeurs.len() * 4);
    for &x in valeurs {
        octets.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&octets)
}

/// Lit exactement `n` f32 little-endian.
fn lire_f32s(r: &mut dyn Read, n: usize) -> std::io::Result<Vec<f32>> {
    let mut octets = vec![0u8; n * 4];
    r.read_exact(&mut octets)?;
    Ok(octets
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn lire_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut octets = [0u8; 4];
    r.read_exact(&mut octets)?;
    Ok(u32::from_le_bytes(octets))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Chemin de fichier temporaire unique pour les tests de sérialisation.
    fn chemin_temporaire(nom: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("echec_nn_test_{}_{}.bin", nom, std::process::id()));
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn nouveau_reseau_tailles_et_sortie_bornee() {
        let net = Mlp::new(1);
        assert_eq!(net.sizes, vec![N_FEATURES, 512, 64, 1]);
        assert_eq!(net.steps, 0);
        let x = vec![0.25f32; N_FEATURES];
        let y = net.forward_one(&x);
        assert!(y.is_finite() && (-1.0..=1.0).contains(&y));
    }

    #[test]
    fn sauvegarde_puis_chargement_exact() {
        let mut net = Mlp::avec_tailles(vec![7, 5, 3, 1], 42);
        // Un pas d'entraînement pour rendre steps et les moments non triviaux.
        let xs: Vec<f32> = (0..2 * 7).map(|i| (i as f32 * 0.37).sin()).collect();
        net.train_batch(&xs, &[0.5, -0.5], 1e-3);

        let chemin = chemin_temporaire("roundtrip");
        net.save(&chemin).expect("échec de la sauvegarde");
        let relu = Mlp::load(&chemin).expect("échec du chargement");
        let _ = std::fs::remove_file(&chemin);

        assert_eq!(relu.sizes, net.sizes);
        assert_eq!(relu.steps, net.steps);
        assert_eq!(relu.weights, net.weights);
        assert_eq!(relu.biases, net.biases);
        assert_eq!(relu.adam_mw, net.adam_mw);
        assert_eq!(relu.adam_vw, net.adam_vw);
        assert_eq!(relu.adam_mb, net.adam_mb);
        assert_eq!(relu.adam_vb, net.adam_vb);
    }

    #[test]
    fn chargement_refuse_mauvais_magic() {
        let chemin = chemin_temporaire("magic");
        std::fs::write(&chemin, b"PASBONNN00000000").unwrap();
        let res = Mlp::load(&chemin);
        let _ = std::fs::remove_file(&chemin);
        assert!(res.is_err());
    }

    #[test]
    fn entrainement_xor_fait_chuter_la_loss() {
        // Petit réseau pour la vitesse : le XOR n'est pas linéairement séparable,
        // la loss ne peut chuter que si la rétropropagation est correcte.
        let mut net = Mlp::avec_tailles(vec![2, 16, 8, 1], 7);
        let xs = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let cibles = [-0.8f32, 0.8, 0.8, -0.8];

        let loss_initiale = net.train_batch(&xs, &cibles, 0.01);
        let mut loss_finale = loss_initiale;
        for _ in 0..300 {
            loss_finale = net.train_batch(&xs, &cibles, 0.01);
        }
        assert!(
            loss_finale < loss_initiale * 0.2,
            "la loss ne chute pas : {loss_initiale} -> {loss_finale}"
        );
        assert_eq!(net.steps, 301);
    }

    #[test]
    fn forward_batch_identique_a_forward_one() {
        let net = Mlp::avec_tailles(vec![10, 8, 4, 1], 3);
        let n = 5;
        let xs: Vec<f32> = (0..n * 10).map(|i| ((i as f32) * 0.61).cos()).collect();
        let lot = net.forward_batch(&xs, n);
        assert_eq!(lot.len(), n);
        for i in 0..n {
            let seul = net.forward_one(&xs[i * 10..(i + 1) * 10]);
            assert_eq!(lot[i], seul, "ligne {i} : lot != unitaire");
        }
    }
}
