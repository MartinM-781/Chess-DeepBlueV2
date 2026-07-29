//! Réseau de valeur : MLP maison (par défaut 773 → 512 → 64 → 1, architecture
//! libre via `new_avec_tailles` — ReLU cachées, tanh en sortie),
//! optimiseur Adam, sérialisation binaire maison (pas de dépendance).
//! `forward_*` doit être utilisable depuis plusieurs threads (&self, aucune mutation).
//!
//! Deux schémas d'entrée cohabitent (`SchemaFeatures`) : le dense historique
//! (773, `features::encode`) et le CREUX roi-zones (6149, `features_roi`) servi
//! par `forward_actifs`/`train_batch_actifs` — la première couche n'y parcourt
//! que les colonnes actives. `evalue_position` route selon le schéma du réseau.
//! Fichiers : `save` écrit le format ECHECNN2 (octet de schéma après les
//! tailles) ; `load` relit AUSSI les anciens fichiers ECHECNN1 (compat totale).

use std::io::{ErrorKind, Read, Write};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rayon::prelude::*;

use shakmaty::Chess;

use crate::features::N_FEATURES;
use crate::features_roi::N_FEATURES_ROI;

/// Hyperparamètres Adam (fixes, seul le taux d'apprentissage est passé en argument).
const ADAM_B1: f32 = 0.9;
const ADAM_B2: f32 = 0.999;
const ADAM_EPS: f32 = 1e-8;

/// Magic des fichiers modèle v1 (8 octets), tout le reste est en little-endian.
/// Format historique SANS schéma de features : toujours accepté en LECTURE
/// (schéma implicite `Classique773`), plus jamais écrit.
const MAGIC: &[u8; 8] = b"ECHECNN1";
/// Magic des fichiers modèle v2 : identique à v1 plus UN octet de schéma de
/// features inséré juste APRÈS les tailles. C'est le format écrit par `save`.
const MAGIC2: &[u8; 8] = b"ECHECNN2";

/// Schéma d'encodage des entrées du réseau.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchemaFeatures {
    /// 773 features pièce-case en perspective du trait (`features::encode`),
    /// chemin DENSE (`forward_one`/`train_batch`).
    Classique773,
    /// 6149 features roi-relatives par zones (`features_roi`), chemin CREUX
    /// (`forward_actifs`/`train_batch_actifs`).
    RoiZones8,
}

impl SchemaFeatures {
    /// Octet de sérialisation du format ECHECNN2.
    fn en_u8(self) -> u8 {
        match self {
            SchemaFeatures::Classique773 => 0,
            SchemaFeatures::RoiZones8 => 1,
        }
    }

    /// Relecture de l'octet de schéma d'un fichier ECHECNN2.
    fn depuis_u8(x: u8) -> Option<SchemaFeatures> {
        match x {
            0 => Some(SchemaFeatures::Classique773),
            1 => Some(SchemaFeatures::RoiZones8),
            _ => None,
        }
    }
}

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
    /// Réseau [N_FEATURES, 512, 64, 1] (architecture par défaut), init He,
    /// graine déterministe.
    pub fn new(seed: u64) -> Self {
        Self::new_avec_tailles(&[N_FEATURES, 512, 64, 1], seed)
    }

    /// Réseau d'architecture ARBITRAIRE [N_FEATURES, c1, …, 1] : même init que
    /// `new` (He, biais et moments Adam à zéro), seules les tailles changent —
    /// la dernière couche reste lue en tanh par `avancer`/`train_batch`.
    /// Sert à élargir le réseau (ex. [773,1024,128,1]) qu'on amorce ensuite
    /// par distillation du champion actuel.
    /// Panique avec un message clair si `tailles[0]` n'est pas N_FEATURES,
    /// s'il y a moins de deux tailles (entrée puis sortie au minimum) ou si la
    /// dernière taille n'est pas 1 : tout le code (`avancer` lit `courant[0]`,
    /// `train_batch` construit un delta par échantillon, l'évaluation
    /// incrémentale) suppose une sortie SCALAIRE — mieux vaut échouer ici
    /// qu'une panique d'indexation différée en plein entraînement.
    pub fn new_avec_tailles(tailles: &[usize], seed: u64) -> Mlp {
        assert!(
            tailles.len() >= 2,
            "new_avec_tailles: il faut au moins deux tailles (entrée puis sortie), reçu {tailles:?}"
        );
        assert_eq!(
            tailles[0], N_FEATURES,
            "new_avec_tailles: la couche d'entrée doit faire N_FEATURES = {N_FEATURES}, reçu {}",
            tailles[0]
        );
        assert_eq!(
            *tailles.last().unwrap(), 1,
            "new_avec_tailles: la dernière couche doit valoir 1 (sortie scalaire tanh), reçu {tailles:?}"
        );
        Self::avec_tailles(tailles.to_vec(), seed)
    }

    /// Réseau au schéma ROI-ZONES (`features_roi`, entrée creuse de
    /// N_FEATURES_ROI = 6149) : même init (He, biais et moments à zéro) et
    /// mêmes gardes que `new_avec_tailles`, mais la couche d'entrée doit faire
    /// N_FEATURES_ROI. Le réseau s'utilise par le CHEMIN CREUX
    /// (`forward_actifs`/`train_batch_actifs`) ; le chemin dense
    /// (`forward_one` sur un vecteur 0/1 de 6149) reste valable et sert de
    /// référence de parité dans les tests.
    pub fn new_roi_zones(tailles: &[usize], seed: u64) -> Mlp {
        assert!(
            tailles.len() >= 2,
            "new_roi_zones: il faut au moins deux tailles (entrée puis sortie), reçu {tailles:?}"
        );
        assert_eq!(
            tailles[0], N_FEATURES_ROI,
            "new_roi_zones: la couche d'entrée doit faire N_FEATURES_ROI = {N_FEATURES_ROI}, reçu {}",
            tailles[0]
        );
        assert_eq!(
            *tailles.last().unwrap(), 1,
            "new_roi_zones: la dernière couche doit valoir 1 (sortie scalaire tanh), reçu {tailles:?}"
        );
        Self::avec_tailles(tailles.to_vec(), seed)
    }

    /// Schéma de features du réseau — DÉRIVÉ de la taille de la couche
    /// d'entrée : N_FEATURES_ROI (6149) → `RoiZones8`, tout le reste →
    /// `Classique773`.
    ///
    /// NOTE DE CHANTIER : le contrat prévoyait un champ stocké
    /// `pub schema: SchemaFeatures` dans `Mlp`. Un champ supplémentaire
    /// casserait les littéraux de structure exhaustifs `Mlp { … }` des tests
    /// de `search.rs` (fichier GELÉ : une autre escouade y travaille en ce
    /// moment) et de `nnue.rs`. Les deux schémas ayant des tailles d'entrée
    /// DISJOINTES (773 ≠ 6149, `new_avec_tailles` et les fichiers v1 imposent
    /// l'un, `new_roi_zones` l'autre), cette dérivation est OBSERVABLEMENT
    /// équivalente au champ. Le format disque v2 stocke bel et bien l'octet de
    /// schéma (contrat respecté côté fichiers) et `load` VÉRIFIE sa cohérence
    /// avec la taille d'entrée. Promouvoir cette méthode en champ sera
    /// mécanique dès que `search.rs` sera dégelé — les appelants écrivent déjà
    /// `net.schema()`.
    pub fn schema(&self) -> SchemaFeatures {
        if self.sizes[0] == N_FEATURES_ROI {
            SchemaFeatures::RoiZones8
        } else {
            SchemaFeatures::Classique773
        }
    }

    /// Construction générique (privée : elle accepte n'importe quelle taille
    /// d'entrée pour les petits réseaux de test ; l'API publique passe par
    /// `new`/`new_avec_tailles` qui imposent N_FEATURES en entrée).
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

    /// Passe avant CREUSE : `actifs` liste les indices d'entrée à 1 (toutes les
    /// autres entrées valent 0, la convention de `features_roi::actifs`). La
    /// première couche ne parcourt que les colonnes actives — pré-activations
    /// = biais + somme des colonnes — puis les couches supérieures sont denses,
    /// mêmes boucles que `avancer` (ReLU cachées, tanh en sortie).
    ///
    /// Équivaut à `forward_one` sur le vecteur 0/1 correspondant (à l'ordre de
    /// sommation flottante près), pour ~32 colonnes lues au lieu de 6149.
    /// Les poids restent au format row-major du chemin dense (l'accès par
    /// colonne est strié) ; l'inférence par deltas, encore plus rapide, est le
    /// rôle des accumulateurs de `nnue.rs`.
    pub fn forward_actifs(&self, actifs: &[u16]) -> f32 {
        let n_couches = self.sizes.len() - 1;
        let (n_in, n1) = (self.sizes[0], self.sizes[1]);
        let cap = self.taille_tampon();
        let mut courant = Vec::with_capacity(cap);
        let mut suivant = Vec::with_capacity(cap);

        // Couche 1 creuse : biais + somme des colonnes actives.
        courant.extend_from_slice(&self.biases[0]);
        let w0 = &self.weights[0];
        for &i in actifs {
            let col = usize::from(i);
            debug_assert!(
                col < n_in,
                "forward_actifs: indice actif {col} hors de la couche d'entrée ({n_in})"
            );
            for j in 0..n1 {
                courant[j] += w0[j * n_in + col];
            }
        }
        if n_couches == 1 {
            return courant[0].tanh();
        }
        for v in courant.iter_mut() {
            *v = v.max(0.0);
        }

        // Couches restantes : denses, mêmes boucles que `avancer`.
        for l in 1..n_couches {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let w = &self.weights[l];
            let b = &self.biases[l];
            suivant.clear();
            suivant.resize(no, 0.0);
            for j in 0..no {
                let ligne = &w[j * ni..(j + 1) * ni];
                let mut s = b[j];
                for k in 0..ni {
                    s += ligne[k] * courant[k];
                }
                suivant[j] = if l + 1 == n_couches { s.tanh() } else { s.max(0.0) };
            }
            std::mem::swap(&mut courant, &mut suivant);
        }
        courant[0]
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

    /// Un pas d'Adam sur un lot CREUX : chaque échantillon est la liste de ses
    /// indices actifs (valeur implicite 1.0, uniques — `features_roi` les
    /// produit ainsi ; des doublons resteraient cohérents, l'entrée comptant
    /// double) et une cible dans (-1, 1). Même loss (MSE après tanh), même
    /// optimiseur et mêmes hyperparamètres que `train_batch` ; la PREMIÈRE
    /// couche ne travaille que sur les colonnes actives, à l'aller (somme de
    /// colonnes) comme au retour : l'activation d'entrée valant 1, le gradient
    /// d'une colonne active est exactement le delta de la couche 1.
    ///
    /// APPROXIMATION ADAM CREUSE (« lazy Adam », standard en embeddings
    /// creux) : les moments m/v et les poids de la couche 1 ne sont mis à jour
    /// QUE pour les colonnes touchées par le lot, et il n'y a PAS de
    /// correction de biais par colonne — les facteurs (1 − β^t) utilisent le
    /// pas GLOBAL `steps`. Conséquences, toutes bénignes : une colonne jamais
    /// active est STRICTEMENT figée (en dense, ses moments hérités
    /// continueraient de décroître et de pousser ses poids quelques pas), et
    /// une colonne rare reçoit une correction calée sur t global plutôt que
    /// sur son propre compteur. Le premier pas depuis un réseau neuf est,
    /// lui, identique au dense (gradient et moments nuls ⇒ mise à jour nulle
    /// sur les colonnes non touchées) — c'est ce que vérifie le test de
    /// parité un-pas.
    pub fn train_batch_actifs(&mut self, lots: &[(Vec<u16>, f32)], lr: f32) -> f32 {
        let n = lots.len();
        assert!(n > 0, "train_batch_actifs: lot vide");
        let n_couches = self.sizes.len() - 1;
        let (n_in, n1) = (self.sizes[0], self.sizes[1]);
        for (actifs, _) in lots {
            for &i in actifs {
                assert!(
                    usize::from(i) < n_in,
                    "train_batch_actifs: indice actif {i} hors de la couche d'entrée ({n_in})"
                );
            }
        }

        // --- Passe avant, en conservant les activations de chaque couche. ---
        // Couche 1 CREUSE, parallèle par échantillon (comme `train_batch`).
        let mut acts: Vec<Vec<f32>> = Vec::with_capacity(n_couches);
        {
            let derniere = n_couches == 1;
            let w0 = &self.weights[0];
            let b0 = &self.biases[0];
            let mut a = vec![0.0f32; n * n1];
            a.par_chunks_mut(n1)
                .zip(lots.par_iter())
                .for_each(|(sortie, (actifs, _))| {
                    sortie.copy_from_slice(b0);
                    for &i in actifs {
                        let col = usize::from(i);
                        for j in 0..n1 {
                            sortie[j] += w0[j * n_in + col];
                        }
                    }
                    for v in sortie.iter_mut() {
                        *v = if derniere { v.tanh() } else { v.max(0.0) };
                    }
                });
            acts.push(a);
        }
        // Couches suivantes : denses, mêmes boucles que `train_batch`.
        for l in 1..n_couches {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            let derniere = l + 1 == n_couches;
            let w = &self.weights[l];
            let b = &self.biases[l];
            let prec = &acts[l - 1];
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

        // --- Loss MSE et delta de sortie (identiques à `train_batch`). ---
        let sorties = &acts[n_couches - 1]; // n × 1
        let mut loss = 0.0f32;
        let mut delta: Vec<f32> = Vec::with_capacity(n);
        for (i, (_, cible)) in lots.iter().enumerate() {
            let y = sorties[i];
            let ecart = y - cible;
            loss += ecart * ecart;
            // dL/dy = 2(y-t)/n ; dy/dz = 1 - tanh² = 1 - y².
            delta.push(2.0 * ecart / n as f32 * (1.0 - y * y));
        }
        loss /= n as f32;

        // --- Rétropropagation + mise à jour Adam couche par couche. ---
        self.steps += 1;
        let c1 = (1.0 - (ADAM_B1 as f64).powi(self.steps as i32)) as f32;
        let c2 = (1.0 - (ADAM_B2 as f64).powi(self.steps as i32)) as f32;

        for l in (0..n_couches).rev() {
            let (ni, no) = (self.sizes[l], self.sizes[l + 1]);
            if l > 0 {
                // Couches denses : copie fidèle de la branche de `train_batch`.
                let prec: &[f32] = &acts[l - 1];
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

                // Delta de la couche précédente (poids AVANT mise à jour),
                // en traversant la ReLU.
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
                adam_maj(&mut self.weights[l], &mut self.adam_mw[l], &mut self.adam_vw[l],
                         &grad_w, lr, c1, c2);
                adam_maj(&mut self.biases[l], &mut self.adam_mb[l], &mut self.adam_vb[l],
                         &grad_b, lr, c1, c2);
                delta = delta_prec;
            } else {
                // --- Couche 1 CREUSE. ---
                debug_assert_eq!(no, n1);
                // Union TRIÉE des colonnes touchées par le lot : parcours et
                // mise à jour déterministes.
                let mut cols: Vec<u16> =
                    lots.iter().flat_map(|(actifs, _)| actifs.iter().copied()).collect();
                cols.sort_unstable();
                cols.dedup();

                // Gradients : biais denses (toujours tous touchés), poids
                // restreints aux colonnes de l'union — gradient d'une colonne
                // active = delta de la couche 1 (activation d'entrée = 1).
                let mut grad_b = vec![0.0f32; no];
                let mut grad_w_cols = vec![0.0f32; cols.len() * no]; // [rang][j]
                for (i, (actifs, _)) in lots.iter().enumerate() {
                    let d = &delta[i * no..(i + 1) * no];
                    for j in 0..no {
                        grad_b[j] += d[j];
                    }
                    for &c in actifs {
                        let rang = cols.binary_search(&c).expect("colonne dans l'union");
                        let g = &mut grad_w_cols[rang * no..(rang + 1) * no];
                        for j in 0..no {
                            g[j] += d[j];
                        }
                    }
                }

                // Adam UNIQUEMENT sur les colonnes touchées, correction de
                // biais GLOBALE c1/c2 (voir l'approximation en tête de doc) —
                // même formule que `adam_maj`, restreinte aux indices striés
                // j * n_in + col.
                let w = &mut self.weights[0];
                let m = &mut self.adam_mw[0];
                let v = &mut self.adam_vw[0];
                for (rang, &c) in cols.iter().enumerate() {
                    let col = usize::from(c);
                    for j in 0..no {
                        let g = grad_w_cols[rang * no + j];
                        let idx = j * n_in + col;
                        m[idx] = ADAM_B1 * m[idx] + (1.0 - ADAM_B1) * g;
                        v[idx] = ADAM_B2 * v[idx] + (1.0 - ADAM_B2) * g * g;
                        let m_chapeau = m[idx] / c1;
                        let v_chapeau = v[idx] / c2;
                        w[idx] -= lr * m_chapeau / (v_chapeau.sqrt() + ADAM_EPS);
                    }
                }
                adam_maj(&mut self.biases[0], &mut self.adam_mb[0], &mut self.adam_vb[0],
                         &grad_b, lr, c1, c2);
            }
        }

        loss
    }

    /// Sérialisation binaire, format v2 : magic "ECHECNN2", sizes, OCTET DE
    /// SCHÉMA (juste après les tailles), steps, poids, biais, moments. Les
    /// fichiers v1 ("ECHECNN1", sans octet de schéma) restent lisibles par
    /// `load` mais ne sont plus produits.
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(MAGIC2)?;
        w.write_all(&(self.sizes.len() as u32).to_le_bytes())?;
        for &s in &self.sizes {
            w.write_all(&(s as u32).to_le_bytes())?;
        }
        w.write_all(&[self.schema().en_u8()])?;
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

    /// Chargement des DEUX formats : "ECHECNN2" (courant, avec octet de
    /// schéma après les tailles) et "ECHECNN1" (historique, schéma implicite
    /// `Classique773`) — compat totale avec tous les modèles déjà sur disque.
    pub fn load(path: &str) -> std::io::Result<Mlp> {
        let mut r = std::io::BufReader::new(std::fs::File::open(path)?);

        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        let v2 = if &magic == MAGIC2 {
            true
        } else if &magic == MAGIC {
            false
        } else {
            return Err(std::io::Error::new(
                ErrorKind::InvalidData,
                "mauvais magic : ni un fichier ECHECNN2 ni un fichier ECHECNN1",
            ));
        };

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

        // v2 : octet de schéma juste après les tailles. Le schéma étant DÉRIVÉ
        // de la taille d'entrée (voir `Mlp::schema`), on VÉRIFIE la cohérence
        // au lieu de stocker l'octet — un fichier incohérent est corrompu.
        if v2 {
            let mut octet = [0u8; 1];
            r.read_exact(&mut octet)?;
            let schema = SchemaFeatures::depuis_u8(octet[0]).ok_or_else(|| {
                std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!("octet de schéma inconnu : {}", octet[0]),
                )
            })?;
            let attendu = if sizes[0] == N_FEATURES_ROI {
                SchemaFeatures::RoiZones8
            } else {
                SchemaFeatures::Classique773
            };
            if schema != attendu {
                return Err(std::io::Error::new(
                    ErrorKind::InvalidData,
                    format!(
                        "schéma {schema:?} incohérent avec la taille d'entrée {}",
                        sizes[0]
                    ),
                ));
            }
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

/// Évaluation d'une position par le réseau, quel que soit son schéma de
/// features : LE point d'entrée commun que tous les sites qui font aujourd'hui
/// `encode` + `forward_one` (bots.rs NetBot, elo, …) devront appeler (chantier
/// de l'escouade Intégration). Route selon `net.schema()` :
/// - `Classique773` : `features::encode` dense dans `tampon` (redimensionné au
///   besoin) puis `forward_one` — bit-à-bit identique au chemin historique ;
/// - `RoiZones8` : `features_roi::actifs` (indices creux, perspective du
///   trait) puis `forward_actifs`. Le petit vecteur d'indices est alloué à
///   chaque appel (≤ 37 u16) ; le chemin chaud de la recherche passera par les
///   accumulateurs de `nnue.rs`, pas par ici.
///
/// Renvoie l'espérance de gain POUR LE TRAIT, dans [-1, 1].
pub fn evalue_position(net: &Mlp, pos: &Chess, tampon: &mut Vec<f32>) -> f32 {
    match net.schema() {
        SchemaFeatures::Classique773 => {
            assert_eq!(
                net.sizes[0], N_FEATURES,
                "evalue_position: schéma Classique773 mais couche d'entrée de {}",
                net.sizes[0]
            );
            tampon.clear();
            tampon.resize(N_FEATURES, 0.0);
            crate::features::encode(pos, tampon);
            net.forward_one(tampon)
        }
        SchemaFeatures::RoiZones8 => {
            let mut actifs: Vec<u16> = Vec::with_capacity(40);
            crate::features_roi::actifs(pos, &mut actifs);
            net.forward_actifs(&actifs)
        }
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
    use shakmaty::Position;

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

    /// new_avec_tailles sur l'architecture ÉLARGIE réelle [773,1024,128,1] :
    /// la loss chute sur un problème jouet (la rétropropagation traverse bien
    /// toutes les couches), puis l'aller-retour disque est EXACT.
    #[test]
    fn new_avec_tailles_apprend_et_serialise_1024_128() {
        let mut net = Mlp::new_avec_tailles(&[N_FEATURES, 1024, 128, 1], 12);
        assert_eq!(net.sizes, vec![N_FEATURES, 1024, 128, 1]);
        assert_eq!(net.steps, 0);

        // Jouet : 4 entrées clairsemées en 0/1 (comme les vraies features),
        // cibles arbitraires dans (-1, 1).
        let n = 4;
        let mut xs = vec![0.0f32; n * N_FEATURES];
        for i in 0..n {
            for k in 0..40 {
                xs[i * N_FEATURES + (i * 131 + k * 19) % N_FEATURES] = 1.0;
            }
        }
        let cibles = [0.7f32, -0.6, 0.4, -0.3];

        // lr 0.001 : à 0.005, Adam (pas ≈ -lr·signe(g) sur ~900 k poids) fait
        // d'abord chuter la loss puis la projette sur un plateau saturé
        // (tanh/ReLU morts) dont elle ne sort plus — mesuré : 0.494 → 0.675.
        // À 0.001, la même config converge (0.494 → 0.001 en 31 pas).
        let loss_initiale = net.train_batch(&xs, &cibles, 0.001);
        let mut loss_finale = loss_initiale;
        for _ in 0..30 {
            loss_finale = net.train_batch(&xs, &cibles, 0.001);
        }
        assert!(
            loss_finale < loss_initiale * 0.5,
            "la loss ne chute pas sur [773,1024,128,1] : {loss_initiale} -> {loss_finale}"
        );
        assert_eq!(net.steps, 31);

        // Aller-retour disque exact (tailles, steps, poids, biais, moments).
        let chemin = chemin_temporaire("elargi_1024");
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

    /// Apprentissage d'une tête PROFONDE (3 couches cachées) : seul test du
    /// crate à exercer `train_batch` au-delà de 2 couches cachées — la
    /// batterie de parité nnue (test 6b, [773,256,64,32,1]) ne couvre ce
    /// chemin que côté INFÉRENCE. Convergence puis aller-retour disque exact.
    #[test]
    fn train_batch_apprend_tete_profonde_32_16_8() {
        let mut net = Mlp::new_avec_tailles(&[N_FEATURES, 32, 16, 8, 1], 9);
        assert_eq!(net.sizes, vec![N_FEATURES, 32, 16, 8, 1]);

        // Mêmes entrées jouets que le test [773,1024,128,1] : clairsemées 0/1.
        let n = 4;
        let mut xs = vec![0.0f32; n * N_FEATURES];
        for i in 0..n {
            for k in 0..40 {
                xs[i * N_FEATURES + (i * 131 + k * 19) % N_FEATURES] = 1.0;
            }
        }
        let cibles = [0.7f32, -0.6, 0.4, -0.3];

        let loss_initiale = net.train_batch(&xs, &cibles, 1e-3);
        let mut loss_finale = loss_initiale;
        for _ in 0..300 {
            loss_finale = net.train_batch(&xs, &cibles, 1e-3);
        }
        assert!(
            loss_finale < loss_initiale * 0.2,
            "la loss ne chute pas sur [773,32,16,8,1] : {loss_initiale} -> {loss_finale}"
        );
        assert_eq!(net.steps, 301);

        // Aller-retour disque exact, moments Adam non triviaux compris.
        let chemin = chemin_temporaire("profond_32_16_8");
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

    /// Garde de new_avec_tailles : une entrée qui n'est pas N_FEATURES panique.
    #[test]
    #[should_panic(expected = "N_FEATURES")]
    fn new_avec_tailles_refuse_mauvaise_entree() {
        let _ = Mlp::new_avec_tailles(&[10, 4, 1], 1);
    }

    /// Garde de new_avec_tailles : une sortie non scalaire panique à la
    /// construction (plutôt qu'une indexation hors bornes en plein
    /// entraînement ou des sorties surnuméraires ignorées en silence).
    #[test]
    #[should_panic(expected = "sortie scalaire")]
    fn new_avec_tailles_refuse_sortie_non_scalaire() {
        let _ = Mlp::new_avec_tailles(&[N_FEATURES, 8, 2], 1);
    }

    /// Garde de new_avec_tailles : moins de deux tailles panique.
    #[test]
    #[should_panic(expected = "deux tailles")]
    fn new_avec_tailles_refuse_moins_de_deux_couches() {
        let _ = Mlp::new_avec_tailles(&[N_FEATURES], 1);
    }

    // =======================================================================
    // Schéma ROI-ZONES : chemin creux, sérialisation v2, compat v1, dispatch.
    // =======================================================================

    /// `n` positions atteintes par coups légaux aléatoires (graine fixe),
    /// en repartant du début quand la partie s'enlise ou se termine.
    fn positions_aleatoires(n: usize, graine: u64) -> Vec<Chess> {
        let mut rng = StdRng::seed_from_u64(graine);
        let mut sortie = Vec::with_capacity(n);
        let mut pos = Chess::default();
        let mut plis = 0usize;
        while sortie.len() < n {
            let coups = pos.legal_moves();
            if coups.is_empty()
                || plis >= 120
                || pos.is_insufficient_material()
                || pos.halfmoves() >= 100
            {
                pos = Chess::default();
                plis = 0;
                continue;
            }
            let coup = coups[rng.gen_range(0..coups.len())].clone();
            pos = pos.play(&coup).expect("coup légal");
            plis += 1;
            sortie.push(pos.clone());
        }
        sortie
    }

    /// Forward dense de RÉFÉRENCE, écrit indépendamment du code de prod : la
    /// matrice pleine de la couche 1 est RECONSTITUÉE colonne par colonne
    /// (le motif d'accès du chemin creux), puis produit matriciel naïf couche
    /// par couche, ReLU cachées et tanh en sortie.
    fn forward_dense_reference(net: &Mlp, x: &[f32]) -> f32 {
        let n_couches = net.sizes.len() - 1;
        let (n_in, n1) = (net.sizes[0], net.sizes[1]);
        let mut w0 = vec![0.0f32; n1 * n_in];
        for col in 0..n_in {
            for j in 0..n1 {
                w0[j * n_in + col] = net.weights[0][j * n_in + col];
            }
        }
        let mut courant: Vec<f32> = x.to_vec();
        for l in 0..n_couches {
            let (ni, no) = (net.sizes[l], net.sizes[l + 1]);
            let w: &[f32] = if l == 0 { &w0 } else { &net.weights[l] };
            let mut suivant = vec![0.0f32; no];
            for j in 0..no {
                let mut s = net.biases[l][j];
                for k in 0..ni {
                    s += w[j * ni + k] * courant[k];
                }
                suivant[j] = if l + 1 == n_couches { s.tanh() } else { s.max(0.0) };
            }
            courant = suivant;
        }
        courant[0]
    }

    /// PARITÉ CREUX vs DENSE : sur 200 positions aléatoires réelles, le
    /// forward creux d'un réseau RoiZones8 jouet doit égaler, à 1e-4, un
    /// forward dense de référence écrit dans le test (matrice pleine
    /// reconstituée depuis les colonnes) ET le `forward_one` de prod sur le
    /// vecteur 0/1 correspondant.
    #[test]
    fn parite_creux_vs_dense_roi_zones_200_positions() {
        let net = Mlp::new_roi_zones(&[N_FEATURES_ROI, 8, 4, 1], 5);
        assert_eq!(net.schema(), SchemaFeatures::RoiZones8);

        let positions = positions_aleatoires(200, 20260729);
        let mut actifs: Vec<u16> = Vec::new();
        let mut x = vec![0.0f32; N_FEATURES_ROI];
        for (idx, pos) in positions.iter().enumerate() {
            crate::features_roi::actifs(pos, &mut actifs);
            let creux = net.forward_actifs(&actifs);

            x.fill(0.0);
            for &i in &actifs {
                x[usize::from(i)] = 1.0;
            }
            let reference = forward_dense_reference(&net, &x);
            assert!(
                (creux - reference).abs() < 1e-4,
                "position {idx} : creux {creux} ≠ référence {reference}"
            );
            let via_forward_one = net.forward_one(&x);
            assert!(
                (creux - via_forward_one).abs() < 1e-4,
                "position {idx} : creux {creux} ≠ forward_one {via_forward_one}"
            );
        }
    }

    /// La loss chute sur un jouet RoiZones8 entraîné par le chemin creux.
    #[test]
    fn train_batch_actifs_fait_chuter_la_loss() {
        let mut net = Mlp::new_roi_zones(&[N_FEATURES_ROI, 16, 1], 7);
        // 6 échantillons clairsemés (~28 indices uniques), cibles dans (-1, 1).
        let cibles = [0.7f32, -0.6, 0.4, -0.3, 0.5, -0.2];
        let lots: Vec<(Vec<u16>, f32)> = cibles
            .iter()
            .enumerate()
            .map(|(i, &cible)| {
                let mut idx: Vec<u16> = (0..28u16)
                    .map(|k| ((i as u16) * 997 + k * 173 + 5) % (N_FEATURES_ROI as u16))
                    .collect();
                idx.sort_unstable();
                idx.dedup();
                (idx, cible)
            })
            .collect();

        let loss_initiale = net.train_batch_actifs(&lots, 1e-3);
        let mut loss_finale = loss_initiale;
        for _ in 0..300 {
            loss_finale = net.train_batch_actifs(&lots, 1e-3);
        }
        assert!(
            loss_finale < loss_initiale * 0.2,
            "la loss ne chute pas : {loss_initiale} -> {loss_finale}"
        );
        assert_eq!(net.steps, 301);
    }

    /// Après UN pas depuis un réseau neuf, le chemin creux et `train_batch`
    /// dense (mêmes entrées en 0/1) doivent coïncider : gradients et moments
    /// nuls sur les colonnes non touchées ⇒ le dense ne les bouge pas non
    /// plus. Un seul échantillon : pas de réduction parallèle multi-
    /// échantillons côté dense, les gradients sont quasi bit-à-bit.
    #[test]
    fn parite_un_pas_creux_vs_dense() {
        let tailles = [N_FEATURES_ROI, 16, 1];
        let mut net_creux = Mlp::new_roi_zones(&tailles, 11);
        let mut net_dense = Mlp::new_roi_zones(&tailles, 11);
        let init = net_creux.weights[0].clone();

        let indices: Vec<u16> = (0..30u16)
            .map(|k| (k * 211 + 17) % (N_FEATURES_ROI as u16))
            .collect();
        let mut x = vec![0.0f32; N_FEATURES_ROI];
        for &i in &indices {
            x[usize::from(i)] = 1.0;
        }
        let cible = 0.6f32;

        let loss_creuse = net_creux.train_batch_actifs(&[(indices.clone(), cible)], 1e-3);
        let loss_dense = net_dense.train_batch(&x, &[cible], 1e-3);
        assert!(
            (loss_creuse - loss_dense).abs() < 1e-6,
            "loss : {loss_creuse} vs {loss_dense}"
        );
        assert_eq!(net_creux.steps, 1);
        assert_eq!(net_dense.steps, 1);

        for l in 0..tailles.len() - 1 {
            for (a, b) in net_creux.weights[l].iter().zip(&net_dense.weights[l]) {
                assert!((a - b).abs() < 1e-4, "couche {l} : poids {a} vs {b}");
            }
            for (a, b) in net_creux.biases[l].iter().zip(&net_dense.biases[l]) {
                assert!((a - b).abs() < 1e-4, "couche {l} : biais {a} vs {b}");
            }
        }

        // Une colonne jamais touchée reste STRICTEMENT figée côté creux.
        let col_libre = (0..N_FEATURES_ROI)
            .find(|c| !indices.contains(&(*c as u16)))
            .unwrap();
        for j in 0..16 {
            let idx = j * N_FEATURES_ROI + col_libre;
            assert_eq!(net_creux.weights[0][idx], init[idx]);
        }
    }

    /// Aller-retour disque v2 d'un réseau RoiZones8 (moments non triviaux),
    /// avec vérification des octets : magic ECHECNN2 et octet de schéma (1)
    /// juste après les tailles.
    #[test]
    fn sauvegarde_v2_roi_zones_aller_retour_exact() {
        let mut net = Mlp::new_roi_zones(&[N_FEATURES_ROI, 8, 1], 42);
        let lots = vec![(vec![3u16, 700, 1600, 6144, 6148], 0.5f32)];
        net.train_batch_actifs(&lots, 1e-3);

        let chemin = chemin_temporaire("v2_roi");
        net.save(&chemin).expect("échec de la sauvegarde");
        let octets = std::fs::read(&chemin).unwrap();
        assert_eq!(&octets[..8], b"ECHECNN2");
        assert_eq!(octets[8 + 4 + 3 * 4], 1); // schéma après les 3 tailles

        let relu = Mlp::load(&chemin).expect("échec du chargement");
        let _ = std::fs::remove_file(&chemin);
        assert_eq!(relu.schema(), SchemaFeatures::RoiZones8);
        assert_eq!(relu.sizes, net.sizes);
        assert_eq!(relu.steps, net.steps);
        assert_eq!(relu.weights, net.weights);
        assert_eq!(relu.biases, net.biases);
        assert_eq!(relu.adam_mw, net.adam_mw);
        assert_eq!(relu.adam_vw, net.adam_vw);
        assert_eq!(relu.adam_mb, net.adam_mb);
        assert_eq!(relu.adam_vb, net.adam_vb);
    }

    /// Un réseau classique sauvé en v2 porte l'octet de schéma 0 et se
    /// recharge en Classique773.
    #[test]
    fn sauvegarde_v2_classique_octet_schema_zero() {
        let net = Mlp::avec_tailles(vec![7, 5, 3, 1], 42);
        let chemin = chemin_temporaire("v2_classique");
        net.save(&chemin).unwrap();
        let octets = std::fs::read(&chemin).unwrap();
        assert_eq!(&octets[..8], b"ECHECNN2");
        assert_eq!(octets[8 + 4 + 4 * 4], 0); // schéma après les 4 tailles

        let relu = Mlp::load(&chemin).unwrap();
        let _ = std::fs::remove_file(&chemin);
        assert_eq!(relu.schema(), SchemaFeatures::Classique773);
        assert_eq!(relu.weights, net.weights);
    }

    /// Écrit `net` au FORMAT v1 (ECHECNN1, sans octet de schéma) — réplique
    /// exacte de l'ancien `save`, pour tester la rétrocompatibilité.
    fn ecrire_fichier_v1(net: &Mlp, chemin: &str) {
        let mut w = std::io::BufWriter::new(std::fs::File::create(chemin).unwrap());
        w.write_all(b"ECHECNN1").unwrap();
        w.write_all(&(net.sizes.len() as u32).to_le_bytes()).unwrap();
        for &s in &net.sizes {
            w.write_all(&(s as u32).to_le_bytes()).unwrap();
        }
        w.write_all(&net.steps.to_le_bytes()).unwrap();
        for groupe in [&net.weights, &net.biases, &net.adam_mw,
                       &net.adam_vw, &net.adam_mb, &net.adam_vb] {
            for couche in groupe {
                for &x in couche {
                    w.write_all(&x.to_le_bytes()).unwrap();
                }
            }
        }
        w.flush().unwrap();
    }

    /// Un fichier v1 synthétique (moments non triviaux) se recharge à
    /// l'identique, avec le schéma implicite Classique773.
    #[test]
    fn chargement_v1_synthetique_schema_classique() {
        let mut net = Mlp::avec_tailles(vec![9, 6, 1], 13);
        let xs: Vec<f32> = (0..2 * 9).map(|i| (i as f32 * 0.31).cos()).collect();
        net.train_batch(&xs, &[0.2, -0.4], 1e-3);

        let chemin = chemin_temporaire("v1_compat");
        ecrire_fichier_v1(&net, &chemin);
        let relu = Mlp::load(&chemin).expect("le format v1 doit rester lisible");
        let _ = std::fs::remove_file(&chemin);
        assert_eq!(relu.schema(), SchemaFeatures::Classique773);
        assert_eq!(relu.sizes, net.sizes);
        assert_eq!(relu.steps, net.steps);
        assert_eq!(relu.weights, net.weights);
        assert_eq!(relu.biases, net.biases);
        assert_eq!(relu.adam_mw, net.adam_mw);
        assert_eq!(relu.adam_vw, net.adam_vw);
        assert_eq!(relu.adam_mb, net.adam_mb);
        assert_eq!(relu.adam_vb, net.adam_vb);
    }

    /// Le palier 1 h RÉEL (models/chess_t1h.bin, format v1, LECTURE SEULE)
    /// doit rester chargeable après le passage de `save` au format v2, et
    /// produire une évaluation finie et bornée.
    #[test]
    fn chargement_du_fichier_v1_reel_chess_t1h() {
        let chemin = "models/chess_t1h.bin";
        assert!(
            std::path::Path::new(chemin).exists(),
            "fichier absent : {chemin} (test lancé hors de la racine du crate ?)"
        );
        let net = Mlp::load(chemin).expect("échec du chargement du modèle v1 réel");
        assert_eq!(net.schema(), SchemaFeatures::Classique773);
        assert_eq!(net.sizes[0], N_FEATURES);
        assert_eq!(*net.sizes.last().unwrap(), 1);
        assert!(net.steps > 0);

        let mut x = vec![0.0f32; N_FEATURES];
        crate::features::encode(&Chess::default(), &mut x);
        let y = net.forward_one(&x);
        assert!(y.is_finite() && (-1.0..=1.0).contains(&y), "évaluation hors bornes : {y}");
    }

    /// Un octet de schéma incohérent avec la taille d'entrée, ou inconnu,
    /// est refusé au chargement.
    #[test]
    fn chargement_v2_refuse_schema_incoherent_ou_inconnu() {
        let net = Mlp::avec_tailles(vec![7, 5, 3, 1], 2);
        let chemin = chemin_temporaire("v2_incoherent");
        net.save(&chemin).unwrap();
        let position_octet = 8 + 4 + 4 * 4;

        let mut octets = std::fs::read(&chemin).unwrap();
        octets[position_octet] = 1; // prétend RoiZones8 avec une entrée de 7
        std::fs::write(&chemin, &octets).unwrap();
        assert!(Mlp::load(&chemin).is_err());

        octets[position_octet] = 9; // schéma inconnu
        std::fs::write(&chemin, &octets).unwrap();
        let res = Mlp::load(&chemin);
        let _ = std::fs::remove_file(&chemin);
        assert!(res.is_err());
    }

    /// `evalue_position` route bien : identique à encode+forward_one pour un
    /// réseau Classique773, identique à actifs+forward_actifs pour RoiZones8.
    #[test]
    fn evalue_position_route_selon_le_schema() {
        let classique = Mlp::new_avec_tailles(&[N_FEATURES, 8, 1], 3);
        let roi = Mlp::new_roi_zones(&[N_FEATURES_ROI, 8, 1], 4);
        assert_eq!(classique.schema(), SchemaFeatures::Classique773);
        assert_eq!(roi.schema(), SchemaFeatures::RoiZones8);

        let mut jeu = positions_aleatoires(5, 99);
        jeu.push(Chess::default());
        let mut tampon = Vec::new();
        for pos in &jeu {
            let mut x = vec![0.0f32; N_FEATURES];
            crate::features::encode(pos, &mut x);
            assert_eq!(evalue_position(&classique, pos, &mut tampon), classique.forward_one(&x));

            let mut a: Vec<u16> = Vec::new();
            crate::features_roi::actifs(pos, &mut a);
            assert_eq!(evalue_position(&roi, pos, &mut tampon), roi.forward_actifs(&a));
        }
    }

    /// Gardes de new_roi_zones : mauvaise entrée, sortie non scalaire,
    /// moins de deux tailles.
    #[test]
    #[should_panic(expected = "N_FEATURES_ROI")]
    fn new_roi_zones_refuse_entree_classique() {
        let _ = Mlp::new_roi_zones(&[N_FEATURES, 8, 1], 1);
    }

    #[test]
    #[should_panic(expected = "sortie scalaire")]
    fn new_roi_zones_refuse_sortie_non_scalaire() {
        let _ = Mlp::new_roi_zones(&[N_FEATURES_ROI, 8, 2], 1);
    }

    #[test]
    #[should_panic(expected = "deux tailles")]
    fn new_roi_zones_refuse_moins_de_deux_tailles() {
        let _ = Mlp::new_roi_zones(&[N_FEATURES_ROI], 1);
    }
}
