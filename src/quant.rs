//! Chemin d'évaluation QUANTIZÉ (entiers 8/16 bits, SIMD AVX2) : le MÊME
//! réseau que `nnue.rs`, lu ~2-3× plus vite par la recherche. L'entraînement
//! reste intégralement en f32 — `QuantNet` est DÉRIVÉ du `Mlp` f32 au
//! chargement (une fois, à la construction du moteur), aucun nouveau format de
//! fichier, aucun changement de sérialisation.
//!
//! SCHÉMA DE QUANTIZATION (échelles dimensionnées sur le champion
//! [773,1024,128,1], mesures du 02/08 : max|w0| = 0.99, max|w1| = 0.66,
//! max|w2| = 0.30, ||W1_ligne||₂ ≤ 3.75, ||W2||₂ = 1.21) :
//!
//! - Couche 0 (entrée → h1) : poids et biais en i16 à l'échelle S0, la plus
//!   grande PUISSANCE DE 2 telle que S0 × max(|w0|,|b0|) ≤ 32767 (capée à
//!   16384 ; champion → S0 = 16384, erreur par poids ≤ 3·10⁻⁵).
//!   ACCUMULATEURS en i32 : deltas incrémentaux EXACTS (aucune saturation
//!   possible, preuve ci-dessous), `depousse` restitue l'étage précédent bit
//!   à bit (troncature de pile, étages indépendants).
//! - Activations : ReLU + clamp en i16 dans [0, ACT_MAX = 8191], échelle
//!   fixe ECHELLE_ACT = 512 (a_int = round(a_f32 × 512), plafond f32 = 16.0).
//!   POURQUOI PAS u8 : avec des activations sur 8 bits (pas P/127, P = plafond
//!   f32), l'écart-type d'arrondi P/(127·√12) se propage par les normes
//!   mesurées ci-dessus en une erreur moyenne ≈ P × 6·10⁻³ sur la sortie
//!   tanh — le seuil « moyenne ≤ 0.01 » imposerait P ≤ 1.2, or ce réseau est
//!   entraîné en f32 SANS CReLU : ses activations dépassent couramment 1.2
//!   (le clip fausserait l'évaluation). Le pas 1/512 rend l'arrondi
//!   d'activation négligeable (~2·10⁻³ sur la sortie) tout en gardant des
//!   POIDS i8 et des produits scalaires AVX2 — c'est la latitude au bâtisseur
//!   du contrat, exercée avec ces mesures à l'appui.
//! - Couches supérieures (têtes) : poids i8 dans [-127, 127], échelle PAR
//!   LIGNE S_j = 127 / max|w_ligne j| (résolution ~2× plus fine que l'échelle
//!   par couche, mesuré : max par ligne médian 0.36 contre 0.66 global).
//!   Produits scalaires en AVX2 : `_mm256_madd_epi16` sur activations i16 ×
//!   poids i8 étendus en i16 (même structure que le `_mm256_maddubs_epi16`
//!   du contrat, largeur halvée — 16 produits par instruction — imposée par
//!   les activations 16 bits justifiées ci-dessus), sommes horizontales en
//!   i32, garde `is_x86_feature_detected!("avx2")` et repli scalaire
//!   STRICTEMENT équivalent bit à bit (tout est entier et sans débordement :
//!   l'ordre des sommations est indifférent — testé).
//! - Sortie : dequantization en f64→f32 puis tanh en f32 — la valeur rendue
//!   reste dans [-1,1] comme le chemin f32.
//!
//! PREUVES D'ABSENCE DE DÉBORDEMENT (les refus sont À LA CONSTRUCTION :
//! `depuis_mlp` rend None pour tout réseau hors domaine, JAMAIS de saturation
//! silencieuse en cours de recherche) :
//! 1. Accumulateur couche 0 (i32) : |qw0| ≤ 32767 et |qb0| ≤ 32767 par
//!    construction de S0 ; un accumulateur = 1 biais + ≤ 32 colonnes de
//!    pièces + ≤ 5 colonnes de drapeaux (à l'évaluation) = ≤ 38 termes, donc
//!    |acc| ≤ 38 × 32767 = 1 245 146 ≪ 2³¹−1 (marge ×1724). Les états
//!    transitoires des deltas (retraits puis ajouts) restent bornés par le
//!    même compte de termes. Arithmétique i32 exacte ⇒ incrémentalité exacte.
//! 2. Produit scalaire des têtes (i32) : n_in ≤ 2048 (refus au-delà),
//!    |a| ≤ 8191, |w| ≤ 127 ⇒ toute somme partielle ≤ 2048 × 8191 × 127
//!    = 2 130 446 336, plus |biais| ≤ 2²⁴ = 16 777 216, total ≤ 2 147 223 552
//!    ≤ 2³¹−1 = 2 147 483 647 (marge 260 095). Production : n_in = 1024,
//!    marge ×2. Les paires de `_mm256_madd_epi16` valent au plus
//!    2 × 8191 × 127 = 2 080 514 ≪ 2³¹ (madd n'a PAS de saturation interne,
//!    contrairement à maddubs — un piège de moins).
//! 3. Requantization par ligne : m_j = round(2³² / S_j) avec S_j ≥ 2 (refus
//!    si max|w_ligne| > 63.5) ⇒ m_j ≤ 2³¹, et |z| ≤ 2³¹ ⇒ |z × m_j| ≤ 2⁶²
//!    < 2⁶³−1 : le produit tient en i64.
//! Chaque borne est doublée d'un `debug_assert!` au point de calcul.
//!
//! PÉRIMÈTRE : les DEUX schémas de features passent (la structure de
//! `PileAccus` se transpose telle quelle : mêmes fonctions d'indexation,
//! mêmes énumérations de deltas, reconstruction roi-zones comprise) ; la
//! cible de production et l'essentiel des batteries visent `Classique773`
//! (le champion), `RoiZones8` étant couvert par une parité réduite.

use shakmaty::{CastlingSide, Chess, Color, EnPassantMode, Move, Position, Role};

use crate::features::N_FEATURES;
use crate::features_roi::{zone_roi, N_FEATURES_ROI};
use crate::nn::{Mlp, SchemaFeatures};
use crate::nnue::{
    indice_piece_roi8, indices_piece, pour_chaque_delta, zones_rois, BASE_DRAPEAUX,
    BASE_DRAPEAUX_ROI8,
};

/// Échelle des activations : a_int = round(a_f32 × 512), pas de 1/512.
const ECHELLE_ACT: i64 = 512;
/// Clamp supérieur des activations (plafond f32 = 8191/512 ≈ 16.0 — jamais
/// approché par le champion, mesuré < 6 ; un réseau qui clipperait ici serait
/// détecté par la batterie de parité).
const ACT_MAX: i32 = 8191;
/// log2(ECHELLE_ACT) : la couche 0 requantize par simple décalage.
const LOG2_ECHELLE_ACT: u32 = 9;
/// Borne du domaine quantizable : tout poids ou biais au-delà (en valeur
/// absolue) fait refuser le réseau (préserve les preuves 1 et 3 ci-dessus).
const MAX_ABS_QUANTIZABLE: f32 = 63.0;
/// Borne des biais quantizés des têtes (préserve la preuve 2).
const MAX_BIAIS_TETE: i64 = 1 << 24;

/// Une couche dense quantizée au-dessus de l'accumulateur.
/// Poids row-major (sortie × entrée) comme `Mlp`, chaque LIGNE à son échelle.
struct CoucheSupQuant {
    n_in: usize,
    n_out: usize,
    /// Poids i8, ligne j = poids[j*n_in .. (j+1)*n_in], échelle s_ligne[j].
    poids: Vec<i8>,
    /// Biais i32 à l'échelle ECHELLE_ACT × s_ligne[j].
    biais: Vec<i32>,
    /// Couches cachées : multiplicateur de requantization par ligne,
    /// m_j = round(2³² / S_j) — act_suivante = (z × m_j + 2³¹) >> 32, clampée.
    /// Vide pour la couche de sortie.
    requant: Vec<i64>,
    /// Couche de sortie : facteur de déquantization par ligne,
    /// 1 / (ECHELLE_ACT × S_j). Vide pour les couches cachées.
    dequant: Vec<f64>,
}

/// Poids du réseau réorganisés et quantizés pour l'évaluation incrémentale
/// entière. Construit UNE FOIS depuis un `Mlp` (source de vérité, jamais
/// modifiée) ; à reconstruire si le réseau change, comme `EvalIncrementale`.
pub struct QuantNet {
    schema: SchemaFeatures,
    h1: usize,
    /// Colonnes i16 de la couche 1, à plat : colonne de la feature f =
    /// `cols[f*h1 .. (f+1)*h1]` (même disposition que `EvalIncrementale`).
    cols: Vec<i16>,
    /// Biais de la couche 1 en i32, échelle S0 (posés dans l'accumulateur).
    biais1: Vec<i32>,
    /// log2(S0 / ECHELLE_ACT) : requantization accumulateur → activation par
    /// décalage arithmétique (S0 est une puissance de 2 ≥ 512).
    decalage0: u32,
    sup: Vec<CoucheSupQuant>,
    /// AVX2 détecté à la construction (garde unique, pas de détection par
    /// évaluation).
    avx2: bool,
}

impl QuantNet {
    /// Dérive le réseau quantizé de `net`. Rend None — refus PROPRE, le
    /// chemin f32 reste disponible — si le réseau est hors domaine :
    /// pas de couche cachée (rien à accélérer), poids ou biais > 63 en valeur
    /// absolue (les preuves anti-débordement ne tiendraient plus), biais de
    /// tête démesuré face aux poids de sa ligne, ou tête plus large que 2048.
    /// Le champion et tous les réseaux d'entraînement du projet sont très
    /// loin de ces bornes (max|w| < 1).
    pub fn depuis_mlp(net: &Mlp) -> Option<QuantNet> {
        if net.sizes.len() < 3 {
            return None; // réseau linéaire : forward complet d'office
        }
        assert_eq!(
            *net.sizes.last().unwrap(),
            1,
            "QuantNet: la dernière couche doit valoir 1 (sortie scalaire tanh), reçu {:?}",
            net.sizes
        );
        let schema = net.schema();
        let n_in = match schema {
            SchemaFeatures::Classique773 => N_FEATURES,
            SchemaFeatures::RoiZones8 => N_FEATURES_ROI,
        };
        assert_eq!(
            net.sizes[0], n_in,
            "QuantNet: couche d'entrée incohérente avec le schéma ({} vs {n_in})",
            net.sizes[0]
        );
        let h1 = net.sizes[1];

        // --- Couche 0 : échelle S0 (puissance de 2), poids i16, biais i32. ---
        // NaN propagé en INFINITY : `f32::max` IGNORE NaN (il rend l'autre
        // opérande), un poids NaN passerait sinon la garde is_finite puis
        // serait quantizé à 0 en release (cast saturant de NaN) — refus
        // exhaustif à la construction, comme les infinis.
        let max_abs0 = net.weights[0]
            .iter()
            .chain(net.biases[0].iter())
            .fold(0.0f32, |m, &v| if v.is_finite() { m.max(v.abs()) } else { f32::INFINITY });
        if !max_abs0.is_finite() || max_abs0 > MAX_ABS_QUANTIZABLE {
            return None;
        }
        // Plus grande puissance de 2 telle que S0 × max_abs0 ≤ 32767, capée à
        // 16384 (le gain de précision au-delà est nul : l'erreur est déjà
        // dominée par l'arrondi d'activation). max_abs0 ≤ 63 garantit
        // S0 ≥ 512 (32767/63 = 520 > 512).
        let mut s0: i64 = 16384;
        while s0 > 512 && (s0 as f32) * max_abs0 > 32767.0 {
            s0 /= 2;
        }
        if (s0 as f32) * max_abs0 > 32767.0 {
            return None; // inatteignable avec la garde max_abs0 ≤ 63, ceinture
        }
        let decalage0 = (s0 as u64).trailing_zeros() - LOG2_ECHELLE_ACT;

        // Transposition + quantization de la couche 1 (disposition identique
        // à EvalIncrementale : une colonne i16 contiguë par feature).
        let w0 = &net.weights[0];
        assert_eq!(w0.len(), h1 * n_in);
        let mut cols = vec![0i16; n_in * h1];
        for j in 0..h1 {
            let ligne = &w0[j * n_in..(j + 1) * n_in];
            for f in 0..n_in {
                let q = (ligne[f] as f64 * s0 as f64).round();
                debug_assert!((-32768.0..=32767.0).contains(&q));
                cols[f * h1 + j] = q as i16;
            }
        }
        let biais1: Vec<i32> = net.biases[0]
            .iter()
            .map(|&b| {
                let q = (b as f64 * s0 as f64).round();
                debug_assert!((-32768.0..=32767.0).contains(&q));
                q as i32
            })
            .collect();

        // --- Têtes : poids i8 à échelle par ligne, biais i32, requant/dequant. ---
        let mut sup = Vec::with_capacity(net.sizes.len() - 2);
        for l in 1..net.sizes.len() - 1 {
            let (ni, no) = (net.sizes[l], net.sizes[l + 1]);
            if ni > 2048 {
                return None; // preuve 2 : n_in ≤ 2048
            }
            let w = &net.weights[l];
            let b = &net.biases[l];
            let derniere = l == net.sizes.len() - 2;
            let mut poids = vec![0i8; ni * no];
            let mut biais = vec![0i32; no];
            let mut requant = Vec::new();
            let mut dequant = Vec::new();
            for j in 0..no {
                let ligne = &w[j * ni..(j + 1) * ni];
                // Même propagation NaN → INFINITY que la couche 0.
                let max_l = ligne
                    .iter()
                    .fold(0.0f32, |m, &v| if v.is_finite() { m.max(v.abs()) } else { f32::INFINITY });
                if !max_l.is_finite() || max_l > MAX_ABS_QUANTIZABLE {
                    return None;
                }
                // Ligne nulle : S_j = 127 (les poids quantizés sont 0 quelle
                // que soit l'échelle ; 127 garde m_j ≤ 2³¹ — preuve 3 — et
                // le biais reste seul porteur du sens, via son échelle).
                let s_j: f64 = if max_l > 0.0 { 127.0 / max_l as f64 } else { 127.0 };
                for (k, &v) in ligne.iter().enumerate() {
                    let q = (v as f64 * s_j).round();
                    debug_assert!((-127.0..=127.0).contains(&q), "poids i8 hors borne");
                    poids[j * ni + k] = q as i8;
                }
                let qb = (b[j] as f64 * ECHELLE_ACT as f64 * s_j).round();
                // !is_finite : un biais NaN rendrait la comparaison > fausse
                // et serait quantizé à 0 — refus, comme les biais démesurés.
                if !qb.is_finite() || qb.abs() > MAX_BIAIS_TETE as f64 {
                    return None; // preuve 2 : |biais| ≤ 2²⁴
                }
                biais[j] = qb as i32;
                if derniere {
                    dequant.push(1.0 / (ECHELLE_ACT as f64 * s_j));
                } else {
                    // Preuve 3 : max_l ≤ 63.5 ⇒ s_j ≥ 2 ⇒ m_j ≤ 2³¹.
                    let m_j = ((1u64 << 32) as f64 / s_j).round() as i64;
                    debug_assert!(m_j <= 1 << 31);
                    requant.push(m_j);
                }
            }
            sup.push(CoucheSupQuant { n_in: ni, n_out: no, poids, biais, requant, dequant });
        }

        Some(QuantNet {
            schema,
            h1,
            cols,
            biais1,
            decalage0,
            sup,
            avx2: avx2_disponible(),
        })
    }

    /// Colonne i16 de la couche 1 associée à la feature `f`.
    #[inline]
    fn colonne(&self, f: usize) -> &[i16] {
        &self.cols[f * self.h1..(f + 1) * self.h1]
    }

    /// Encode complètement `pos` dans les DEUX perspectives : racine de la
    /// pile d'accumulateurs entière (miroir exact de `EvalIncrementale::racine`).
    pub fn racine(&self, pos: &Chess) -> PileQuant {
        if self.schema == SchemaFeatures::RoiZones8 {
            return self.racine_roi8(pos);
        }
        let h1 = self.h1;
        // Réserve pour ~128 plis de recherche sans réallocation.
        let mut donnees = Vec::with_capacity(2 * h1 * 128);
        donnees.extend(self.biais1.iter().copied());
        donnees.extend(self.biais1.iter().copied());
        {
            let (blanc, noir) = donnees.split_at_mut(h1);
            for (case, piece) in pos.board().iter() {
                let (ib, inoir) = indices_piece(piece.color, piece.role, case);
                accumule_i32(blanc, self.colonne(ib), 1);
                accumule_i32(noir, self.colonne(inoir), 1);
            }
        }
        PileQuant { donnees, h1 }
    }

    /// `racine` du schéma roi-zones : chaque perspective conditionnée par la
    /// zone de SON PROPRE roi (miroir de `EvalIncrementale::racine_roi8`).
    fn racine_roi8(&self, pos: &Chess) -> PileQuant {
        let h1 = self.h1;
        let mut donnees = Vec::with_capacity(2 * h1 * 128);
        donnees.extend(self.biais1.iter().copied());
        donnees.extend(self.biais1.iter().copied());
        {
            let (blanc, noir) = donnees.split_at_mut(h1);
            let (zone_blanche, zone_noire) = zones_rois(pos);
            for (case, piece) in pos.board().iter() {
                let ib = indice_piece_roi8(piece.color, piece.role, case, true, zone_blanche);
                let inoir = indice_piece_roi8(piece.color, piece.role, case, false, zone_noire);
                accumule_i32(blanc, self.colonne(ib), 1);
                accumule_i32(noir, self.colonne(inoir), 1);
            }
        }
        PileQuant { donnees, h1 }
    }
}

/// AVX2 présent sur cette machine ? (Garde du contrat ; sur toute autre
/// architecture que x86_64 le repli scalaire est le seul chemin.)
fn avx2_disponible() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// `dst += signe × col` (col i16 élargie en i32). Exact : voir preuve 1 —
/// aucune somme d'au plus 38 termes bornés par 32767 ne peut déborder i32.
#[inline]
fn accumule_i32(dst: &mut [i32], col: &[i16], signe: i32) {
    debug_assert_eq!(dst.len(), col.len());
    // Boucle simple : l'auto-vectorisation AVX2 (target-cpu=native) la
    // convertit en additions entières par paquets de 8.
    for (d, c) in dst.iter_mut().zip(col) {
        *d += signe * i32::from(*c);
    }
}

/// Produit scalaire entier activations i16 × poids i8, REPLI SCALAIRE.
/// Somme en i32 : exact (preuve 2), donc identique bit à bit au chemin AVX2
/// quel que soit l'ordre des sommations.
#[inline]
fn produit_scalaire_scalaire(acts: &[i16], poids: &[i8]) -> i32 {
    debug_assert_eq!(acts.len(), poids.len());
    let mut somme = 0i32;
    for (a, w) in acts.iter().zip(poids) {
        somme += i32::from(*a) * i32::from(*w);
    }
    somme
}

/// Produit scalaire entier en AVX2 : 16 produits i16×i16 par
/// `_mm256_madd_epi16` (poids i8 élargis par `_mm256_cvtepi8_epi16`), sommes
/// partielles en 8 voies i32, somme horizontale finale, queue scalaire.
/// Chaque paire vaut au plus 2 × 8191 × 127 = 2 080 514 (pas de saturation
/// interne dans madd) et la somme totale est bornée par la preuve 2 :
/// résultat exact, bit à bit égal au repli scalaire.
///
/// # Safety
/// Ne doit être appelée que si AVX2 est disponible (garde `QuantNet::avx2`,
/// détectée par `is_x86_feature_detected` à la construction).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn produit_scalaire_avx2(acts: &[i16], poids: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    debug_assert_eq!(acts.len(), poids.len());
    let n = acts.len();
    let mut somme = _mm256_setzero_si256();
    let mut k = 0usize;
    while k + 16 <= n {
        let a = _mm256_loadu_si256(acts.as_ptr().add(k) as *const __m256i);
        let w8 = _mm_loadu_si128(poids.as_ptr().add(k) as *const __m128i);
        let w = _mm256_cvtepi8_epi16(w8);
        somme = _mm256_add_epi32(somme, _mm256_madd_epi16(a, w));
        k += 16;
    }
    let mut voies = [0i32; 8];
    _mm256_storeu_si256(voies.as_mut_ptr() as *mut __m256i, somme);
    let mut total: i32 = voies.iter().sum();
    while k < n {
        total += i32::from(*acts.get_unchecked(k)) * i32::from(*poids.get_unchecked(k));
        k += 1;
    }
    total
}

/// Produit scalaire : AVX2 si disponible, repli scalaire sinon — les deux
/// chemins rendent le même i32 exact.
#[inline]
fn produit_scalaire(avx2: bool, acts: &[i16], poids: &[i8]) -> i32 {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SÛRETÉ : `avx2` vient de is_x86_feature_detected à la construction.
        return unsafe { produit_scalaire_avx2(acts, poids) };
    }
    let _ = avx2;
    produit_scalaire_scalaire(acts, poids)
}

/// Pile d'accumulateurs ENTIERS de la couche 1, un étage par pli — protocole
/// identique à `PileAccus` (pousse / pousse_null / depousse / evalue), la
/// recherche pilote l'une ou l'autre selon `utilise_int8`.
pub struct PileQuant {
    /// Étages concaténés : l'étage k occupe `donnees[k*2*h1 .. (k+1)*2*h1]`,
    /// [perspective blanche | perspective noire].
    donnees: Vec<i32>,
    h1: usize,
}

impl PileQuant {
    /// Tranche du sommet de pile (2×h1 valeurs).
    #[inline]
    fn base_sommet(&self) -> usize {
        self.donnees.len() - 2 * self.h1
    }

    /// Empile la position atteinte en jouant `m` depuis `pos_avant` — mêmes
    /// deltas que `PileAccus::pousse`, en arithmétique i32 exacte.
    pub fn pousse(&mut self, quant: &QuantNet, pos_avant: &Chess, m: &Move) {
        debug_assert_eq!(self.h1, quant.h1, "pousse: QuantNet d'une autre taille");
        if quant.schema == SchemaFeatures::RoiZones8 {
            return self.pousse_roi8(quant, pos_avant, m);
        }
        let h1 = self.h1;
        let base = self.base_sommet();
        // Duplique le sommet : le nouvel étage part de la position courante.
        self.donnees.extend_from_within(base..);
        let sommet = self.donnees.len() - 2 * h1;
        let (blanc, noir) = self.donnees[sommet..].split_at_mut(h1);

        let nous = pos_avant.turn();
        pour_chaque_delta(nous, m, |couleur, role, case, signe| {
            let (ib, inoir) = indices_piece(couleur, role, case);
            accumule_i32(blanc, quant.colonne(ib), signe as i32);
            accumule_i32(noir, quant.colonne(inoir), signe as i32);
        });
    }

    /// `pousse` du schéma roi-zones : deltas dans les deux perspectives tant
    /// que les rois restent dans leurs zones, RECONSTRUCTION de la perspective
    /// dont le roi traverse une frontière (transcription exacte de
    /// `PileAccus::pousse_roi8` en entier).
    fn pousse_roi8(&mut self, quant: &QuantNet, pos_avant: &Chess, m: &Move) {
        let h1 = self.h1;
        let base = self.base_sommet();
        self.donnees.extend_from_within(base..);
        let sommet = self.donnees.len() - 2 * h1;

        let nous = pos_avant.turn();
        let nous_blanc = nous == Color::White;
        let (zone_blanche, zone_noire) = zones_rois(pos_avant);

        // Case d'arrivée de NOTRE roi s'il bouge (coup ordinaire ou roque).
        let arrivee_roi = match m {
            Move::Normal { role: Role::King, to, .. } => Some(*to),
            Move::Castle { .. } => {
                let cote = m.castling_side().expect("Move::Castle a toujours un côté");
                Some(cote.king_to(nous))
            }
            _ => None,
        };
        let zone_apres = arrivee_roi.and_then(|case| {
            let (avant, apres) = if nous_blanc {
                (zone_blanche, zone_roi(usize::from(case)))
            } else {
                (zone_noire, zone_roi(usize::from(case) ^ 56))
            };
            (apres != avant).then_some(apres)
        });

        let (blanc, noir) = self.donnees[sommet..].split_at_mut(h1);
        match zone_apres {
            None => pour_chaque_delta(nous, m, |couleur, role, case, signe| {
                let ib = indice_piece_roi8(couleur, role, case, true, zone_blanche);
                let inoir = indice_piece_roi8(couleur, role, case, false, zone_noire);
                accumule_i32(blanc, quant.colonne(ib), signe as i32);
                accumule_i32(noir, quant.colonne(inoir), signe as i32);
            }),
            Some(zone) => {
                let mut apres = pos_avant.clone();
                apres.play_unchecked(m);
                let (reconstruit, garde, zone_gardee) = if nous_blanc {
                    (blanc, noir, zone_noire)
                } else {
                    (noir, blanc, zone_blanche)
                };
                reconstruit.copy_from_slice(&quant.biais1);
                for (case, piece) in apres.board().iter() {
                    let idx = indice_piece_roi8(piece.color, piece.role, case, nous_blanc, zone);
                    accumule_i32(reconstruit, quant.colonne(idx), 1);
                }
                pour_chaque_delta(nous, m, |couleur, role, case, signe| {
                    let idx = indice_piece_roi8(couleur, role, case, !nous_blanc, zone_gardee);
                    accumule_i32(garde, quant.colonne(idx), signe as i32);
                });
            }
        }
    }

    /// Null-move : sommet dupliqué, les perspectives s'échangent à la lecture.
    pub fn pousse_null(&mut self) {
        let base = self.base_sommet();
        self.donnees.extend_from_within(base..);
    }

    /// Dépile un étage — troncature : l'étage précédent est restitué BIT À
    /// BIT (les étages sont des copies indépendantes, l'arithmétique entière
    /// n'a pas laissé de trace).
    pub fn depousse(&mut self) {
        assert!(
            self.donnees.len() >= 4 * self.h1,
            "depousse: la racine de la pile ne peut pas être dépilée"
        );
        let nouvelle_taille = self.donnees.len() - 2 * self.h1;
        self.donnees.truncate(nouvelle_taille);
    }

    /// Évalue la position du sommet de pile (`pos` DOIT être cette position) :
    /// accumulateur de la perspective du trait + colonnes des drapeaux actifs,
    /// requantization ReLU→i16, têtes i8 (AVX2 ou repli scalaire), sortie
    /// déquantizée puis tanh f32. Écart au chemin f32 borné par la batterie de
    /// parité (max ≤ 0.05, moyenne ≤ 0.01 exigés).
    pub fn evalue(&self, quant: &QuantNet, pos: &Chess) -> f32 {
        debug_assert_eq!(self.h1, quant.h1, "evalue: QuantNet d'une autre taille");
        let h1 = self.h1;
        let base = self.base_sommet();
        let nous = pos.turn();
        let sommet = &self.donnees[base..];
        let accu = if nous == Color::White { &sommet[..h1] } else { &sommet[h1..] };

        // Copie de travail : le sommet de pile ne doit pas être modifié.
        let mut courant: Vec<i32> = accu.to_vec();

        // Drapeaux non incrémentaux (≤ 5 colonnes), mêmes conditions que le
        // chemin f32 — la preuve 1 compte ces 5 termes.
        let base_drapeaux = match quant.schema {
            SchemaFeatures::Classique773 => BASE_DRAPEAUX,
            SchemaFeatures::RoiZones8 => BASE_DRAPEAUX_ROI8,
        };
        let eux = nous.other();
        let roques = pos.castles();
        if roques.has(nous, CastlingSide::KingSide) {
            accumule_i32(&mut courant, quant.colonne(base_drapeaux), 1);
        }
        if roques.has(nous, CastlingSide::QueenSide) {
            accumule_i32(&mut courant, quant.colonne(base_drapeaux + 1), 1);
        }
        if roques.has(eux, CastlingSide::KingSide) {
            accumule_i32(&mut courant, quant.colonne(base_drapeaux + 2), 1);
        }
        if roques.has(eux, CastlingSide::QueenSide) {
            accumule_i32(&mut courant, quant.colonne(base_drapeaux + 3), 1);
        }
        if pos.ep_square(EnPassantMode::Legal).is_some() {
            accumule_i32(&mut courant, quant.colonne(base_drapeaux + 4), 1);
        }

        // ReLU + requantization par décalage : acc (échelle S0) → activation
        // i16 (échelle 512), arrondi au plus proche, clamp [0, ACT_MAX].
        // (décalage0 = 0 ⟺ S0 = 512 : clamp direct.)
        let t0 = quant.decalage0;
        let mut acts: Vec<i16> = Vec::with_capacity(h1);
        if t0 == 0 {
            acts.extend(courant.iter().map(|&v| v.clamp(0, ACT_MAX) as i16));
        } else {
            let arrondi = 1i32 << (t0 - 1);
            acts.extend(
                courant
                    .iter()
                    .map(|&v| ((v + arrondi) >> t0).clamp(0, ACT_MAX) as i16),
            );
        }

        // Têtes : produits scalaires entiers, requantization par ligne sur les
        // couches cachées, déquantization + tanh sur la sortie.
        let n_sup = quant.sup.len();
        let mut suivant: Vec<i16> = Vec::new();
        for (l, couche) in quant.sup.iter().enumerate() {
            debug_assert_eq!(acts.len(), couche.n_in);
            let derniere = l + 1 == n_sup;
            if derniere {
                debug_assert_eq!(couche.n_out, 1);
                let ligne = &couche.poids[..couche.n_in];
                let z = produit_scalaire(quant.avx2, &acts, ligne) + couche.biais[0];
                let y = (f64::from(z) * couche.dequant[0]) as f32;
                return y.tanh();
            }
            suivant.clear();
            suivant.reserve(couche.n_out);
            for j in 0..couche.n_out {
                let ligne = &couche.poids[j * couche.n_in..(j + 1) * couche.n_in];
                let z = produit_scalaire(quant.avx2, &acts, ligne) + couche.biais[j];
                // Preuve 3 : |z| ≤ 2³¹ et m_j ≤ 2³¹ ⇒ le produit tient en i64.
                let m_j = couche.requant[j];
                let a = ((i64::from(z) * m_j + (1i64 << 31)) >> 32).clamp(0, i64::from(ACT_MAX));
                suivant.push(a as i16);
            }
            std::mem::swap(&mut acts, &mut suivant);
        }
        unreachable!("QuantNet a toujours au moins une couche de tête");
    }
}

// ---------------------------------------------------------------------------
// Tests. Deux étages :
// 1. tests RAPIDES (cargo test --lib) : équivalence bit à bit SIMD/scalaire,
//    refus des réseaux hors domaine, parité statique et incrémentale sur
//    petits réseaux (parties aléatoires + scripts roques/promotions/e.p.),
//    null-move, invariant « deltas ≡ reconstruction » EXACT (arithmétique
//    entière), schéma roi-zones réduit ;
// 2. batteries #[ignore] (release) : parité int8 vs f32 sur ≥ 50 000
//    positions (parties aléatoires + livre + finales, seuils du contrat
//    max ≤ 0.05 / moyenne ≤ 0.01) et bench nœuds/s f32 vs int8.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bots::{Bot, RandomBot};
    use crate::nn::evalue_position;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use shakmaty::fen::Fen;
    use shakmaty::uci::UciMove;
    use shakmaty::{CastlingMode, FromSetup};
    use std::sync::Arc;

    /// Seuils du contrat sur |int8 − f32| (sorties tanh dans [-1,1]).
    const SEUIL_MAX: f32 = 0.05;
    const SEUIL_MOYENNE: f32 = 0.01;

    fn pos_de_fen(fen: &str) -> Chess {
        fen.parse::<Fen>()
            .expect("FEN invalide")
            .into_position(CastlingMode::Standard)
            .expect("position illégale")
    }

    /// Référence f32 : le répartiteur de schéma de production (encode +
    /// forward_one en Classique773, chemin creux en roi-zones).
    fn reference(net: &Mlp, pos: &Chess) -> f32 {
        let mut tampon = Vec::new();
        evalue_position(net, pos, &mut tampon)
    }

    /// Petit Mlp Classique773 aléatoire [773, h1, h2, 1] aux BIAIS NON NULS
    /// (même précaution que les tests de nnue.rs : une perte des biais doit
    /// se voir), étroit pour rester rapide en debug.
    fn petit_reseau(graine: u64, h1: usize, h2: usize) -> Mlp {
        let mut net = Mlp::new_avec_tailles(&[N_FEATURES, h1, h2, 1], graine);
        let mut rng = StdRng::seed_from_u64(graine ^ 0x0A11E5);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        net
    }

    /// Petit réseau roi-zones (biais non nuls, mêmes raisons).
    fn petit_reseau_roi8(graine: u64, h1: usize, h2: usize) -> Mlp {
        let mut net = Mlp::new_roi_zones(&[N_FEATURES_ROI, h1, h2, 1], graine);
        let mut rng = StdRng::seed_from_u64(graine ^ 0x0A11E5);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        net
    }

    /// Partie RandomBot depuis `depart` : liste (position AVANT coup, coup).
    fn partie_aleatoire(depart: &Chess, graine: u64, max_plis: usize) -> Vec<(Chess, Move)> {
        let mut bot = RandomBot::new(graine);
        let mut pos = depart.clone();
        let mut partie = Vec::new();
        for _ in 0..max_plis {
            if pos.is_insufficient_material() || pos.halfmoves() >= 100 {
                break;
            }
            let m = match bot.choose(&pos) {
                Some(m) => m,
                None => break, // mat ou pat
            };
            let suivante = pos.clone().play(&m).expect("coup légal");
            partie.push((pos, m));
            pos = suivante;
        }
        partie
    }

    /// Partie scriptée en notation UCI depuis `depart`.
    fn construit_partie(depart: &Chess, ucis: &[&str]) -> Vec<(Chess, Move)> {
        let mut pos = depart.clone();
        let mut partie = Vec::new();
        for u in ucis {
            let m = UciMove::from_ascii(u.as_bytes())
                .expect("UCI invalide")
                .to_move(&pos)
                .expect("coup illégal dans la partie construite");
            let suivante = pos.clone().play(&m).expect("coup légal");
            partie.push((pos, m));
            pos = suivante;
        }
        partie
    }

    /// Même plateau, trait inversé (test du null-move) ; None si illégal.
    fn inverse_trait(pos: &Chess) -> Option<Chess> {
        let mut setup = pos.clone().into_setup(EnPassantMode::Legal);
        setup.turn = !setup.turn;
        setup.ep_square = None;
        Chess::from_setup(setup, CastlingMode::Standard).ok()
    }

    /// Rejoue `partie` en maintenant la pile quantizée : à CHAQUE coup,
    /// écart borné contre la référence f32 ET égalité BIT À BIT avec une
    /// racine reconstruite de zéro (l'arithmétique entière rend les deltas
    /// exactement équivalents à la reconstruction — aucune tolérance).
    fn verifie_partie(net: &Mlp, quant: &QuantNet, partie: &[(Chess, Move)], contexte: &str) {
        if partie.is_empty() {
            return;
        }
        let mut pile = quant.racine(&partie[0].0);
        for (i, (avant, m)) in partie.iter().enumerate() {
            pile.pousse(quant, avant, m);
            let apres = avant.clone().play(m).expect("coup légal");
            let attendu = reference(net, &apres);
            let obtenu = pile.evalue(quant, &apres);
            assert!(
                (attendu - obtenu).abs() <= SEUIL_MAX,
                "{contexte}, coup {i} ({m:?}) : int8 {obtenu} vs f32 {attendu}"
            );
            let reconstruit = quant.racine(&apres).evalue(quant, &apres);
            assert_eq!(
                obtenu, reconstruit,
                "{contexte}, coup {i} : deltas ≠ reconstruction (accumulateurs divergés)"
            );
        }
    }

    /// 1. SIMD = scalaire, bit à bit : vecteurs aléatoires aux bornes du
    /// domaine (|a| ≤ ACT_MAX, |w| ≤ 127), longueurs avec et sans queue
    /// (multiples de 16 ou non), plus les pires cas tout-aux-bornes.
    #[test]
    fn produit_scalaire_avx2_egal_scalaire() {
        if !avx2_disponible() {
            println!("AVX2 indisponible : repli scalaire seul, test sans objet");
            return;
        }
        let mut rng = StdRng::seed_from_u64(0x51D);
        for &n in &[0usize, 1, 7, 15, 16, 17, 31, 32, 100, 128, 1000, 1024, 2048] {
            for essai in 0..8 {
                let acts: Vec<i16> = (0..n).map(|_| rng.gen_range(0..=ACT_MAX) as i16).collect();
                let poids: Vec<i8> = (0..n).map(|_| rng.gen_range(-127..=127i32) as i8).collect();
                let scalaire = produit_scalaire_scalaire(&acts, &poids);
                let simd = unsafe { produit_scalaire_avx2(&acts, &poids) };
                assert_eq!(scalaire, simd, "n={n}, essai {essai}");
            }
        }
        // Pire cas de la preuve 2 : 2048 termes tous aux bornes.
        let acts = vec![ACT_MAX as i16; 2048];
        for w in [-127i8, 127] {
            let poids = vec![w; 2048];
            let scalaire = produit_scalaire_scalaire(&acts, &poids);
            let simd = unsafe { produit_scalaire_avx2(&acts, &poids) };
            assert_eq!(scalaire, simd, "pire cas w={w}");
            assert_eq!(scalaire, 2048 * ACT_MAX * i32::from(w), "somme pire cas exacte");
        }
    }

    /// 2. Refus PROPRE des réseaux hors domaine : linéaire (rien à accélérer),
    /// poids géant (les preuves anti-débordement ne tiendraient plus), biais
    /// démesuré face aux poids de sa ligne. Un réseau ordinaire passe.
    #[test]
    fn refus_des_reseaux_hors_domaine() {
        assert!(QuantNet::depuis_mlp(&petit_reseau(1, 16, 8)).is_some());

        // Réseau linéaire [773,1] : pas de couche cachée.
        let lineaire = Mlp::new_avec_tailles(&[N_FEATURES, 1], 2);
        assert!(QuantNet::depuis_mlp(&lineaire).is_none());

        // Poids de couche 0 hors domaine (> 63).
        let mut gros = petit_reseau(3, 16, 8);
        gros.weights[0][123] = 100.0;
        assert!(QuantNet::depuis_mlp(&gros).is_none());

        // Poids de tête hors domaine.
        let mut gros_tete = petit_reseau(4, 16, 8);
        gros_tete.weights[1][7] = 80.0;
        assert!(QuantNet::depuis_mlp(&gros_tete).is_none());

        // Biais de tête démesuré face aux poids de sa ligne : l'échelle de la
        // ligne enverrait le biais quantizé au-delà de 2^24.
        let mut biais_fou = petit_reseau(5, 16, 8);
        for w in biais_fou.weights[1][..16].iter_mut() {
            *w = 1e-4;
        }
        biais_fou.biases[1][0] = 50.0;
        assert!(QuantNet::depuis_mlp(&biais_fou).is_none());

        // NaN : refus PROPRE sur les trois gardes (fold couche 0, fold de
        // ligne de tête, biais de tête) — sans la propagation, f32::max
        // ignorerait NaN et le poids serait quantizé à 0 en release.
        let mut nan_c0 = petit_reseau(8, 16, 8);
        nan_c0.weights[0][42] = f32::NAN;
        assert!(QuantNet::depuis_mlp(&nan_c0).is_none());
        let mut nan_tete = petit_reseau(9, 16, 8);
        nan_tete.weights[1][5] = f32::NAN;
        assert!(QuantNet::depuis_mlp(&nan_tete).is_none());
        let mut nan_biais = petit_reseau(10, 16, 8);
        nan_biais.biases[1][0] = f32::NAN;
        assert!(QuantNet::depuis_mlp(&nan_biais).is_none());

        // Ligne de tête ENTIÈREMENT NULLE (neurone mort) : accepté — échelle
        // de repli 127, requant dans les bornes — et parité conservée.
        let mut ligne_nulle = petit_reseau(6, 16, 8);
        for w in ligne_nulle.weights[1][3 * 16..4 * 16].iter_mut() {
            *w = 0.0;
        }
        let quant = QuantNet::depuis_mlp(&ligne_nulle).expect("ligne nulle acceptée");
        let pos = Chess::default();
        let ecart =
            (reference(&ligne_nulle, &pos) - quant.racine(&pos).evalue(&quant, &pos)).abs();
        assert!(ecart <= SEUIL_MAX, "parité avec ligne nulle : écart {ecart}");
    }

    /// 2 bis. Le CHAMPION de production est dans le domaine de quantization
    /// (chargement toléré en échec : train.exe peut réécrire le fichier).
    #[test]
    fn champion_derivable() {
        let chemin = concat!(env!("CARGO_MANIFEST_DIR"), "/models/chess_best.bin");
        match Mlp::load(chemin) {
            Ok(net) => {
                let q = QuantNet::depuis_mlp(&net);
                assert!(
                    q.is_some(),
                    "le champion {chemin} (tailles {:?}) doit être quantizable",
                    net.sizes
                );
                println!(
                    "champion {:?} schéma {:?} : quantizé (décalage0 = {})",
                    net.sizes,
                    net.schema(),
                    q.unwrap().decalage0
                );
            }
            Err(e) => println!("{chemin} illisible ({e}) : test sauté"),
        }
    }

    /// 3. Parité statique : 150 positions de parties aléatoires, racine() +
    /// evalue() contre le forward f32, seuil du contrat.
    #[test]
    fn parite_statique_petit_reseau() {
        let net = petit_reseau(7, 32, 12);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau quantizable");
        let mut positions = vec![Chess::default()];
        let mut graine = 0u64;
        while positions.len() < 150 {
            for (pos, _) in partie_aleatoire(&Chess::default(), 900 + graine, 90) {
                positions.push(pos);
                if positions.len() >= 150 {
                    break;
                }
            }
            graine += 1;
        }
        let mut somme = 0.0f64;
        let mut max = 0.0f32;
        for (i, pos) in positions.iter().enumerate() {
            let attendu = reference(&net, pos);
            let obtenu = quant.racine(pos).evalue(&quant, pos);
            let ecart = (attendu - obtenu).abs();
            somme += f64::from(ecart);
            max = max.max(ecart);
            assert!(ecart <= SEUIL_MAX, "position {i} : int8 {obtenu} vs f32 {attendu}");
        }
        let moyenne = (somme / positions.len() as f64) as f32;
        println!("parité statique : max {max:.5}, moyenne {moyenne:.5}");
        assert!(moyenne <= SEUIL_MOYENNE, "moyenne {moyenne} > {SEUIL_MOYENNE}");
    }

    /// 4. Parité incrémentale : 12 parties aléatoires rejouées coup à coup —
    /// écart borné à chaque coup ET égalité exacte deltas/reconstruction.
    #[test]
    fn parite_incrementale_parties_aleatoires() {
        let net = petit_reseau(11, 32, 12);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau quantizable");
        for g in 0..12u64 {
            let partie = partie_aleatoire(&Chess::default(), 4000 + g, 120);
            verifie_partie(&net, &quant, &partie, &format!("partie aléatoire {g}"));
        }
    }

    /// 5. Scripts déterministes : les quatre roques, promotions avec et sans
    /// capture (deux camps), prises en passant des deux camps — les mêmes
    /// scripts que la batterie de nnue.rs, la couverture ne dépend d'aucun aléa.
    #[test]
    fn parite_roques_promotions_en_passant() {
        let net = petit_reseau(13, 24, 8);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau quantizable");
        let scripts: [(&str, Vec<&str>, &str); 5] = [
            (
                "initiale",
                vec!["e2e4", "d7d5", "g1f3", "b8c6", "f1c4", "c8f5", "e1g1", "d8d6", "d2d3", "e8c8"],
                "O-O blanc / O-O-O noir",
            ),
            (
                "initiale",
                vec!["d2d4", "e7e5", "c1e3", "f8e7", "b1c3", "g8f6", "d1d2", "e8g8", "e1c1"],
                "O-O-O blanc / O-O noir",
            ),
            (
                "rnbqkb1r/ppppppPp/8/8/8/8/PPPPPPpP/RNBQKB1R w KQkq - 0 1",
                vec!["g7h8q", "g2h1n"],
                "promotions avec capture",
            ),
            (
                "8/4k1P1/8/8/8/8/6p1/4K3 w - - 0 1",
                vec!["g7g8q", "g2g1n"],
                "promotions calmes",
            ),
            (
                "initiale",
                vec![
                    "e2e4", "g8f6", "e4e5", "d7d5", "e5d6", "c7d6", "g1f3", "b7b5", "f3g1",
                    "b5b4", "c2c4", "b4c3",
                ],
                "prises en passant des deux camps",
            ),
        ];
        let (mut roques, mut promotions, mut en_passants) = (0usize, 0usize, 0usize);
        for (fen, ucis, contexte) in &scripts {
            let depart = if *fen == "initiale" { Chess::default() } else { pos_de_fen(fen) };
            let partie = construit_partie(&depart, ucis);
            for (_, m) in &partie {
                if m.is_castle() {
                    roques += 1;
                }
                if m.promotion().is_some() {
                    promotions += 1;
                }
                if m.is_en_passant() {
                    en_passants += 1;
                }
            }
            verifie_partie(&net, &quant, &partie, contexte);
        }
        assert_eq!((roques, promotions, en_passants), (4, 4, 2), "couverture des scripts");
    }

    /// 6. Null-move : accumulateurs inchangés, l'évaluation lit l'autre
    /// perspective ; parité au trait inversé et retour EXACT après depousse.
    #[test]
    fn pousse_null_echange_les_perspectives() {
        let net = petit_reseau(17, 24, 8);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau quantizable");
        let mut testes = 0;
        for g in 0..6u64 {
            let partie = partie_aleatoire(&Chess::default(), 300 + g, 60);
            if partie.is_empty() {
                continue;
            }
            let mut pile = quant.racine(&partie[0].0);
            for (avant, m) in &partie {
                pile.pousse(&quant, avant, m);
                let apres = avant.clone().play(m).expect("coup légal");
                if let Some(inverse) = inverse_trait(&apres) {
                    let avant_null = pile.evalue(&quant, &apres);
                    pile.pousse_null();
                    let obtenu = pile.evalue(&quant, &inverse);
                    let attendu = reference(&net, &inverse);
                    assert!(
                        (attendu - obtenu).abs() <= SEUIL_MAX,
                        "null-move en partie {g} : {obtenu} vs {attendu}"
                    );
                    pile.depousse();
                    assert_eq!(
                        pile.evalue(&quant, &apres),
                        avant_null,
                        "depousse du null-move ne restitue pas l'évaluation"
                    );
                    testes += 1;
                }
            }
        }
        assert!(testes > 20, "trop peu de null-moves testés ({testes})");
    }

    /// 7. Marche pousse/depousse (60 %/40 %) : parité et égalité exacte
    /// deltas/reconstruction après CHAQUE pas — les étages dépilés (y compris
    /// sous null-move) ne laissent aucune trace, par construction entière.
    #[test]
    fn depousse_rejoint_la_reference() {
        let net = petit_reseau(19, 24, 8);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau quantizable");
        let mut rng = StdRng::seed_from_u64(77);
        let mut pile = quant.racine(&Chess::default());
        let mut positions = vec![Chess::default()];
        for pas in 0..800 {
            let sommet = positions.last().unwrap().clone();
            let coups = sommet.legal_moves();
            let pousser = !coups.is_empty() && (positions.len() == 1 || rng.gen_bool(0.6));
            if pousser {
                let m = coups[rng.gen_range(0..coups.len())].clone();
                pile.pousse(&quant, &sommet, &m);
                positions.push(sommet.play(&m).expect("coup légal"));
            } else if positions.len() > 1 {
                pile.depousse();
                positions.pop();
            } else {
                break;
            }
            let pos = positions.last().unwrap();
            let attendu = reference(&net, pos);
            let obtenu = pile.evalue(&quant, pos);
            assert!(
                (attendu - obtenu).abs() <= SEUIL_MAX,
                "pas {pas} (profondeur {}) : {obtenu} vs {attendu}",
                positions.len()
            );
            assert_eq!(
                obtenu,
                quant.racine(pos).evalue(&quant, pos),
                "pas {pas} : deltas ≠ reconstruction"
            );
        }
    }

    /// Dépiler la racine est un bug de l'appelant : panique attendue.
    #[test]
    #[should_panic(expected = "depousse")]
    fn depousse_sous_la_racine_panique() {
        let net = petit_reseau(23, 8, 4);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau quantizable");
        let mut pile = quant.racine(&Chess::default());
        pile.depousse();
    }

    /// 8. Schéma ROI-ZONES (parité réduite — le chemin est la transcription
    /// exacte de PileAccus, exercée ici sur LE cas critique : marches de roi
    /// à travers les frontières de zones, reconstruction comprise).
    #[test]
    fn parite_roi8_marches_de_roi() {
        let net = petit_reseau_roi8(31, 24, 8);
        let quant = QuantNet::depuis_mlp(&net).expect("réseau roi8 quantizable");

        // Marche des deux rois de nnue.rs (12 traversées de frontières).
        let marche = construit_partie(&pos_de_fen("7k/8/8/8/8/8/8/K7 w - - 0 1"), &[
            "a1a2", "h8h7", "a2a3", "h7h6", "a3a4", "h6h5", "a4a5", "h5h4",
            "a5b5", "h4g4", "b5b6", "g4g3", "b6b7", "g3g2", "b7c7", "g2g1",
            "c7c6", "g1f1", "c6d6", "f1e1", "d6e6", "e1d1", "e6e5", "d1c1",
            "e5e4", "c1b1", "e4d4", "b1b2", "d4d3", "b2a2", "d3d2", "a2a1",
        ]);
        verifie_partie(&net, &quant, &marche, "marche des deux rois (roi8)");

        // Capture PAR le roi en traversant une frontière, puis redescentes.
        let capture = construit_partie(&pos_de_fen("8/b4k2/K7/8/8/8/1P6/8 w - - 0 1"), &[
            "a6a7", "f7e6", "a7b6", "e6d5", "b6b5", "d5e4", "b5b4", "e4f3",
        ]);
        verifie_partie(&net, &quant, &capture, "capture par le roi (roi8)");

        // Parties aléatoires courtes (roques, e.p. selon l'aléa) + grands
        // roques scriptés (changement de zone garanti).
        for g in 0..4u64 {
            let partie = partie_aleatoire(&Chess::default(), 8100 + g, 80);
            verifie_partie(&net, &quant, &partie, &format!("partie roi8 {g}"));
        }
        let grands_roques = construit_partie(&Chess::default(), &[
            "d2d4", "d7d5", "c1e3", "c8e6", "b1c3", "b8c6", "d1d2", "d8d7",
            "e1c1", "e8c8",
        ]);
        verifie_partie(&net, &quant, &grands_roques, "grands roques (roi8)");
    }

    /// 9. La RECHERCHE en int8 : rend un coup, un score dans l'échelle
    /// attendue, est déterministe à budget de nœuds fixe, et exerce la garde
    /// débug échantillonnée de search::evaluer (premier nœud de chaque
    /// recherche). Le drapeau OFF du même chercheur repasse au f32 (les deux
    /// voies rendent des coups — pas nécessairement identiques, l'évaluation
    /// diffère de ~1e-3).
    #[test]
    fn recherche_int8_deterministe_et_bornee() {
        use crate::search::{Limites, Recherche, SCORE_MAT};
        let net = Arc::new(petit_reseau(41, 24, 8));
        let pos = pos_de_fen("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4");
        let limites = Limites { max_noeuds: 3000, max_profondeur: 0, movetime_ms: 0 };

        let mut r1 = Recherche::new(net.clone(), 14);
        r1.utilise_int8 = true;
        let a = r1.cherche(&pos, limites);
        assert!(a.coup.is_some(), "un coup légal existe");
        assert!(
            a.score.abs() <= 1.0 + 1e-6 || a.score.abs() > SCORE_MAT - 200.0,
            "score hors échelle : {}",
            a.score
        );

        // Déterminisme bit à bit à budget fixe (chercheur neuf).
        let mut r2 = Recherche::new(net.clone(), 14);
        r2.utilise_int8 = true;
        let b = r2.cherche(&pos, limites);
        assert_eq!(a.coup, b.coup);
        assert_eq!(a.score, b.score);
        assert_eq!(a.noeuds, b.noeuds);

        // Drapeau coupé sur le MÊME chercheur : retour au chemin f32.
        r1.utilise_int8 = false;
        let c = r1.cherche(&pos, limites);
        assert!(c.coup.is_some());
    }

    // -----------------------------------------------------------------------
    // Batteries #[ignore] — pensées pour le release.
    //
    // Parité (contrat) :
    //   cargo test --release --lib quant::tests::parite_int8 -- --ignored --nocapture
    //   (volume par défaut 50 000 positions ; fumé : QUANT_PARITE_POSITIONS=2000)
    // Bench :
    //   cargo test --release --lib quant::tests::bench_int8 -- --ignored --nocapture
    //   (QUANT_BENCH_NOEUDS=..., QUANT_BENCH_POSITIONS=3 pour un fumé)
    // -----------------------------------------------------------------------

    /// Réseau des batteries : le CHAMPION de production (chess_best.bin), à
    /// défaut chess_latest.bin, à défaut un aléatoire aux tailles réelles
    /// [773,1024,128,1] (chargements tolérés en échec : train.exe écrit).
    fn reseau_batterie() -> (Arc<Mlp>, String) {
        for nom in ["chess_best.bin", "chess_latest.bin"] {
            let chemin = format!("{}/models/{nom}", env!("CARGO_MANIFEST_DIR"));
            match Mlp::load(&chemin) {
                Ok(net) if net.schema() == SchemaFeatures::Classique773 => {
                    println!("batterie : réseau réel {nom} (tailles {:?})", net.sizes);
                    return (Arc::new(net), format!("réseau réel {nom}"));
                }
                Ok(net) => println!("batterie : {nom} au schéma {:?}, ignoré", net.schema()),
                Err(e) => println!("batterie : {nom} illisible ({e})"),
            }
        }
        let mut net = Mlp::new_avec_tailles(&[N_FEATURES, 1024, 128, 1], 0xACE);
        let mut rng = StdRng::seed_from_u64(0xB1A15);
        for biais in net.biases.iter_mut() {
            for b in biais.iter_mut() {
                *b = rng.gen::<f32>() * 0.2 - 0.1;
            }
        }
        println!("batterie : repli sur un réseau aléatoire [773,1024,128,1]");
        (Arc::new(net), "réseau aléatoire [773,1024,128,1]".to_string())
    }

    /// Volume de la batterie de parité (positions évaluées), réglable pour un
    /// fumé : QUANT_PARITE_POSITIONS=2000.
    fn nb_positions_parite() -> usize {
        std::env::var("QUANT_PARITE_POSITIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(50_000)
    }

    /// Accumulateur de statistiques d'écart |int8 − f32|.
    #[derive(Default)]
    struct StatsEcart {
        n: usize,
        somme: f64,
        max: f32,
        /// Bornes : 1e-3, 5e-3, 1e-2, 2e-2, 5e-2, +∞.
        buckets: [usize; 6],
    }

    impl StatsEcart {
        fn pousse(&mut self, ecart: f32) {
            self.n += 1;
            self.somme += f64::from(ecart);
            self.max = self.max.max(ecart);
            let bornes = [1e-3f32, 5e-3, 1e-2, 2e-2, 5e-2];
            let idx = bornes.iter().position(|&b| ecart <= b).unwrap_or(5);
            self.buckets[idx] += 1;
        }

        fn moyenne(&self) -> f32 {
            if self.n == 0 { 0.0 } else { (self.somme / self.n as f64) as f32 }
        }

        fn affiche(&self, contexte: &str) {
            println!(
                "{contexte} : {} évals, max {:.5}, moyenne {:.6} | ≤1e-3 {} | ≤5e-3 {} | \
                 ≤1e-2 {} | ≤2e-2 {} | ≤5e-2 {} | >5e-2 {}",
                self.n, self.max, self.moyenne(),
                self.buckets[0], self.buckets[1], self.buckets[2],
                self.buckets[3], self.buckets[4], self.buckets[5],
            );
        }
    }

    /// BATTERIE DE PARITÉ (contrat) : évaluation int8 contre f32 sur
    /// ≥ 50 000 positions variées — parties aléatoires depuis la position
    /// initiale (~40 %), depuis le livre d'ouvertures (~30 %) et depuis des
    /// finales générées (~30 %) — pile incrémentale maintenue coup à coup,
    /// reconstruction vérifiée EXACTE une position sur 16. Rapport
    /// max / moyenne / distribution, échec si max > 0.05 ou moyenne > 0.01.
    #[test]
    #[ignore]
    fn parite_int8_contre_f32_50000_positions() {
        let objectif = nb_positions_parite();
        let (net, nom) = reseau_batterie();
        let quant = QuantNet::depuis_mlp(&net).expect("réseau des batteries quantizable");

        let mut global = StatsEcart::default();
        let mut par_source: [StatsEcart; 3] = Default::default();
        let noms_sources = ["parties aléatoires", "livre", "finales"];
        let mut rng_departs = StdRng::seed_from_u64(0xDE9A47);
        let mut g = 0u64;
        while global.n < objectif {
            // ~40 % initiale, ~30 % livre, ~30 % finales, en alternance fixe.
            let source = match g % 10 {
                0..=3 => 0usize,
                4..=6 => 1,
                _ => 2,
            };
            let depart = match source {
                0 => Chess::default(),
                1 => crate::departs::tirage(&mut rng_departs, 1.0, 0.0).pos,
                _ => crate::departs::tirage(&mut rng_departs, 0.0, 1.0).pos,
            };
            let partie = partie_aleatoire(&depart, 0x1247_0000 + g, 120);
            g += 1;
            if partie.is_empty() {
                continue;
            }
            let mut pile = quant.racine(&partie[0].0);
            for (i, (avant, m)) in partie.iter().enumerate() {
                pile.pousse(&quant, avant, m);
                let apres = avant.clone().play(m).expect("coup légal");
                let obtenu = pile.evalue(&quant, &apres);
                let attendu = reference(&net, &apres);
                let ecart = (attendu - obtenu).abs();
                global.pousse(ecart);
                par_source[source].pousse(ecart);
                // Invariant entier : deltas ≡ reconstruction, bit à bit.
                if i % 16 == 0 {
                    assert_eq!(
                        obtenu,
                        quant.racine(&apres).evalue(&quant, &apres),
                        "[{nom}] partie {g}, coup {i} : deltas ≠ reconstruction"
                    );
                }
                if global.n >= objectif {
                    break;
                }
            }
        }

        for (s, stats) in par_source.iter().enumerate() {
            stats.affiche(&format!("  {}", noms_sources[s]));
        }
        global.affiche(&format!("[{nom}] parité int8/f32"));
        assert!(
            global.max <= SEUIL_MAX,
            "[{nom}] max |int8 − f32| = {} > {SEUIL_MAX}",
            global.max
        );
        assert!(
            global.moyenne() <= SEUIL_MOYENNE,
            "[{nom}] moyenne |int8 − f32| = {} > {SEUIL_MOYENNE}",
            global.moyenne()
        );
    }

    /// BENCH (contrat) : nœuds/s de la recherche f32 contre int8, même budget
    /// de nœuds, TT de même taille, sur un jeu FIXE de positions de milieu de
    /// partie. QUANT_BENCH_NOEUDS (défaut 200 000) et QUANT_BENCH_POSITIONS
    /// (défaut 6 ; fumé : 3) règlent le volume.
    ///
    /// MÉTHODE : les deux bras sont ENTRELACÉS par position, dans un ordre
    /// alterné (f32/int8, int8/f32, ...) pour ne pas imputer une dérive de la
    /// machine à un seul bras. Le chiffre de RÉFÉRENCE se mesure MACHINE AU
    /// REPOS : sous la charge permanente train.exe/serve.exe, la contention
    /// mémoire pénalise davantage le chemin f32 (~4× plus d'octets lus par
    /// évaluation : tête 1024×128 f32 = 512 Ko contre 131 Ko i8) et GONFLE le
    /// ratio (mesuré ×12-13 sous charge, contre les ~×3-4 attendus à vide).
    #[test]
    #[ignore]
    fn bench_int8_contre_f32_noeuds_par_seconde() {
        use crate::search::{Limites, Recherche};

        // Même exemption EcoQoS que les main() des binaires : sans elle,
        // Windows peut confiner le processus de test sur les cœurs efficients.
        crate::pleine_puissance();

        let budget: u64 = std::env::var("QUANT_BENCH_NOEUDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(200_000);
        let n_positions: usize = std::env::var("QUANT_BENCH_POSITIONS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6);
        let (net, nom) = reseau_batterie();

        // Positions de milieu de partie : 3 FEN fixes déjà utilisées par les
        // tests du projet + des milieux atteints par 16 plis aléatoires
        // (graines fixes : le jeu est identique d'un lancement à l'autre).
        let mut positions = vec![
            pos_de_fen("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 3 9"),
            pos_de_fen("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            pos_de_fen("r3k2r/pppq1ppp/2npbn2/2b1p3/2B1P3/2NPBN2/PPPQ1PPP/R3K2R w KQkq - 0 1"),
        ];
        let mut graine = 0u64;
        while positions.len() < n_positions {
            let partie = partie_aleatoire(&Chess::default(), 0xBE7C_0000 + graine, 16);
            graine += 1;
            if partie.len() == 16 {
                positions.push(partie[15].0.clone().play(&partie[15].1).expect("coup légal"));
            }
        }
        positions.truncate(n_positions);

        let limites = Limites { max_noeuds: budget, max_profondeur: 0, movetime_ms: 0 };
        // (noeuds, secondes) : [0] = f32, [1] = int8. Bras entrelacés par
        // position, ordre alterné A/B, B/A, ... (voir MÉTHODE ci-dessus).
        let mut bilans = [(0u64, 0.0f64), (0u64, 0.0f64)];
        for (i, pos) in positions.iter().enumerate() {
            let ordre: [usize; 2] = if i % 2 == 0 { [0, 1] } else { [1, 0] };
            for mode in ordre {
                let int8 = mode == 1;
                let mut r = Recherche::new(net.clone(), 20);
                r.utilise_int8 = int8;
                let debut = std::time::Instant::now();
                let res = r.cherche(pos, limites);
                let secondes = debut.elapsed().as_secs_f64();
                println!(
                    "  {} position {i} : {} nœuds, {:.0} ms, {:.0} nœuds/s, prof {}",
                    if int8 { "int8" } else { "f32 " },
                    res.noeuds,
                    secondes * 1e3,
                    res.noeuds as f64 / secondes,
                    res.profondeur
                );
                bilans[mode].0 += res.noeuds;
                bilans[mode].1 += secondes;
            }
        }
        let vitesse_f32 = bilans[0].0 as f64 / bilans[0].1;
        let vitesse_int8 = bilans[1].0 as f64 / bilans[1].1;
        println!(
            "[{nom}] bench ({} positions, {budget} nœuds/position) : \
             f32 {vitesse_f32:.0} nœuds/s, int8 {vitesse_int8:.0} nœuds/s, ratio ×{:.2}",
            positions.len(),
            vitesse_int8 / vitesse_f32
        );
        println!(
            "NB : chiffre de référence machine AU REPOS — sous la charge \
             train.exe/serve.exe, la contention mémoire gonfle le ratio."
        );
    }
}
