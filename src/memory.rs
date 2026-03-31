//! Memory retrieval — ACT-R declarative memory, activation, retrieval.
//!
//! Implements Anderson's ACT-R base-level learning equation, spreading
//! activation, retrieval probability (softmax gate), and retrieval latency.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite, validate_positive};

/// A memory chunk's access history for ACT-R base-level computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkHistory {
    /// Times since each prior presentation (in seconds, all positive).
    pub presentation_ages: Vec<f64>,
}

/// ACT-R base-level activation (Anderson & Lebiere, 1998).
///
/// `B_i = ln( Σ t_j^(-d) )`
///
/// where `t_j` is the time since the j-th presentation and `d` is the
/// decay parameter (typically 0.5).
///
/// Higher activation = more accessible memory.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any presentation age is
/// non-positive, decay is non-positive, or the history is empty.
#[must_use = "returns the base-level activation without side effects"]
pub fn base_level_activation(history: &ChunkHistory, decay: f64) -> Result<f64> {
    if history.presentation_ages.is_empty() {
        return Err(BodhError::InvalidParameter(
            "presentation_ages must not be empty".into(),
        ));
    }
    validate_positive(decay, "decay")?;

    let mut sum = 0.0;
    for (i, &t) in history.presentation_ages.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 {
            return Err(BodhError::InvalidParameter(format!(
                "presentation_ages[{i}] must be positive, got {t}"
            )));
        }
        sum += t.powf(-decay);
    }
    Ok(sum.ln())
}

/// Spreading activation from associated sources.
///
/// `S_i = B_i + Σ (W_j × S_ji)`
///
/// where `W_j` is the attentional weight on source j and `S_ji` is the
/// associative strength between source j and chunk i.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[must_use = "returns the total activation without side effects"]
pub fn spreading_activation(base_level: f64, associations: &[(f64, f64)]) -> Result<f64> {
    validate_finite(base_level, "base_level")?;

    let mut spread = 0.0;
    for (i, &(weight, strength)) in associations.iter().enumerate() {
        if !weight.is_finite() {
            return Err(BodhError::InvalidParameter(format!(
                "weight[{i}] must be finite, got {weight}"
            )));
        }
        if !strength.is_finite() {
            return Err(BodhError::InvalidParameter(format!(
                "strength[{i}] must be finite, got {strength}"
            )));
        }
        spread += weight * strength;
    }
    Ok(base_level + spread)
}

/// ACT-R retrieval probability (softmax retrieval gate).
///
/// `P(retrieve) = 1 / (1 + e^((τ − A_i) / s))`
///
/// where `A_i` is the chunk's activation, `τ` (tau) is the retrieval
/// threshold, and `s` is the noise parameter (activation noise ≈ 0.4).
///
/// When `A_i > τ`, probability approaches 1.
/// When `A_i < τ`, probability approaches 0.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or s is non-positive.
#[inline]
#[must_use = "returns the retrieval probability without side effects"]
pub fn retrieval_probability(activation: f64, threshold: f64, noise: f64) -> Result<f64> {
    validate_finite(activation, "activation")?;
    validate_finite(threshold, "threshold")?;
    validate_positive(noise, "noise")?;
    let exponent = (threshold - activation) / noise;
    Ok(1.0 / (1.0 + exponent.exp()))
}

/// ACT-R retrieval latency.
///
/// `T = F × e^(−f × A_i)`
///
/// where `F` is the latency factor (typically ~1.0 s) and `f` is the
/// latency exponent (typically ~1.0). Higher activation → faster retrieval.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if `latency_factor` or `latency_exponent`
/// is non-positive, or activation is non-finite.
#[inline]
#[must_use = "returns the retrieval time in seconds without side effects"]
pub fn retrieval_latency(
    activation: f64,
    latency_factor: f64,
    latency_exponent: f64,
) -> Result<f64> {
    validate_finite(activation, "activation")?;
    validate_positive(latency_factor, "latency_factor")?;
    validate_positive(latency_exponent, "latency_exponent")?;
    Ok(latency_factor * (-latency_exponent * activation).exp())
}

/// Partial matching penalty for a chunk that partially matches a retrieval cue.
///
/// `MP = Σ P_k × similarity(desired_k, actual_k)`
///
/// where `P_k` is the mismatch penalty weight for slot k and
/// `similarity` is in \[-1, 0\] (0 = perfect match, -1 = maximum mismatch).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any value is non-finite or
/// similarity is outside \[-1, 0\].
#[must_use = "returns the partial matching penalty without side effects"]
pub fn partial_matching(slot_penalties: &[(f64, f64)]) -> Result<f64> {
    let mut total = 0.0;
    for (i, &(penalty, similarity)) in slot_penalties.iter().enumerate() {
        if !penalty.is_finite() {
            return Err(BodhError::InvalidParameter(format!(
                "penalty[{i}] must be finite, got {penalty}"
            )));
        }
        if !similarity.is_finite() || !(-1.0..=0.0).contains(&similarity) {
            return Err(BodhError::InvalidParameter(format!(
                "similarity[{i}] must be in [-1, 0], got {similarity}"
            )));
        }
        total += penalty * similarity;
    }
    Ok(total)
}

// ---------------------------------------------------------------------------
// Levels of Processing (Craik & Lockhart, 1972)
// ---------------------------------------------------------------------------

/// Processing depth from Craik & Lockhart's levels-of-processing framework.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ProcessingLevel {
    /// Structural/physical features (shallowest).
    Structural,
    /// Phonological/acoustic features.
    Phonological,
    /// Semantic/meaning-based features (deepest).
    Semantic,
}

impl ProcessingLevel {
    /// Encoding strength multiplier for this processing level.
    ///
    /// Deeper processing produces stronger, more durable memory traces.
    /// Based on typical levels-of-processing effect sizes.
    #[inline]
    #[must_use]
    pub fn encoding_strength(self) -> f64 {
        match self {
            Self::Structural => 0.3,
            Self::Phonological => 0.5,
            Self::Semantic => 1.0,
        }
    }
}

/// Encoding strength: memory trace strength as a function of
/// processing depth and elaboration.
///
/// `strength = level_strength × (1 + elaboration) × distinctiveness`
///
/// where `elaboration` (0+) captures the richness of encoding
/// (number of associations formed) and `distinctiveness` (0–1)
/// captures how unique the encoding is relative to other items.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[inline]
#[must_use = "returns the encoding strength without side effects"]
pub fn encoding_strength(
    level: ProcessingLevel,
    elaboration: f64,
    distinctiveness: f64,
) -> Result<f64> {
    validate_finite(elaboration, "elaboration")?;
    validate_finite(distinctiveness, "distinctiveness")?;
    let elab = elaboration.max(0.0);
    let dist = distinctiveness.clamp(0.0, 1.0);
    Ok(level.encoding_strength() * (1.0 + elab) * dist)
}

/// Generation effect: items self-generated during encoding are
/// remembered better than items simply read.
///
/// `retention_boost = base_retention × (1 + generation_weight × generated)`
///
/// where `generated` is 1.0 if the item was self-generated and 0.0
/// if passively read. Typical generation effect ≈ 15–20% boost.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[inline]
#[must_use = "returns the boosted retention without side effects"]
pub fn generation_effect(
    base_retention: f64,
    generated: bool,
    generation_weight: f64,
) -> Result<f64> {
    validate_finite(base_retention, "base_retention")?;
    validate_finite(generation_weight, "generation_weight")?;
    let boost = if generated { generation_weight } else { 0.0 };
    Ok((base_retention * (1.0 + boost)).clamp(0.0, 1.0))
}

/// Testing effect (retrieval practice): retrieving information
/// strengthens memory more than restudying.
///
/// `new_strength = old_strength + retrieval_bonus × success × difficulty`
///
/// where `success` is 1.0 for successful retrieval (0.0 for failure),
/// and `difficulty` (0–1) modulates the bonus (harder retrievals
/// produce stronger strengthening — desirable difficulty).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[inline]
#[must_use = "returns the updated strength without side effects"]
pub fn testing_effect(
    old_strength: f64,
    retrieval_bonus: f64,
    success: bool,
    difficulty: f64,
) -> Result<f64> {
    validate_finite(old_strength, "old_strength")?;
    validate_finite(retrieval_bonus, "retrieval_bonus")?;
    validate_finite(difficulty, "difficulty")?;
    let diff = difficulty.clamp(0.0, 1.0);
    let bonus = if success { retrieval_bonus * diff } else { 0.0 };
    Ok(old_strength + bonus)
}

/// Encoding specificity (Tulving & Thomson, 1973): retrieval
/// probability depends on match between encoding and retrieval contexts.
///
/// `P(recall) = base_probability × context_match^specificity`
///
/// where `context_match` (0–1) is the overlap between encoding and
/// retrieval contexts, and `specificity` controls how sharply
/// performance drops with mismatch (typically 1.0–3.0).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or
/// `specificity` is non-positive.
#[inline]
#[must_use = "returns the recall probability without side effects"]
pub fn encoding_specificity(
    base_probability: f64,
    context_match: f64,
    specificity: f64,
) -> Result<f64> {
    validate_finite(base_probability, "base_probability")?;
    validate_finite(context_match, "context_match")?;
    validate_positive(specificity, "specificity")?;
    let cm = context_match.clamp(0.0, 1.0);
    Ok((base_probability * cm.powf(specificity)).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Base-level activation --

    #[test]
    fn test_base_level_single_recent() {
        // One presentation 1s ago, d=0.5: B = ln(1^-0.5) = ln(1) = 0.
        let h = ChunkHistory {
            presentation_ages: vec![1.0],
        };
        let b = base_level_activation(&h, 0.5).unwrap();
        assert!(b.abs() < 1e-10);
    }

    #[test]
    fn test_base_level_multiple_presentations() {
        // More presentations → higher activation.
        let h1 = ChunkHistory {
            presentation_ages: vec![1.0],
        };
        let h5 = ChunkHistory {
            presentation_ages: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        let b1 = base_level_activation(&h1, 0.5).unwrap();
        let b5 = base_level_activation(&h5, 0.5).unwrap();
        assert!(b5 > b1);
    }

    #[test]
    fn test_base_level_recency() {
        // More recent → higher activation.
        let recent = ChunkHistory {
            presentation_ages: vec![1.0],
        };
        let old = ChunkHistory {
            presentation_ages: vec![100.0],
        };
        let b_recent = base_level_activation(&recent, 0.5).unwrap();
        let b_old = base_level_activation(&old, 0.5).unwrap();
        assert!(b_recent > b_old);
    }

    #[test]
    fn test_base_level_empty() {
        let h = ChunkHistory {
            presentation_ages: vec![],
        };
        assert!(base_level_activation(&h, 0.5).is_err());
    }

    #[test]
    fn test_base_level_invalid_age() {
        let h = ChunkHistory {
            presentation_ages: vec![1.0, -1.0],
        };
        assert!(base_level_activation(&h, 0.5).is_err());
    }

    #[test]
    fn test_base_level_known_value() {
        // Single presentation at t=4, d=0.5: B = ln(4^-0.5) = ln(0.5) = -0.693...
        let h = ChunkHistory {
            presentation_ages: vec![4.0],
        };
        let b = base_level_activation(&h, 0.5).unwrap();
        assert!((b - 0.5_f64.ln()).abs() < 1e-10);
    }

    // -- Spreading activation --

    #[test]
    fn test_spreading_no_associations() {
        let total = spreading_activation(1.0, &[]).unwrap();
        assert!((total - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spreading_adds_to_base() {
        let assocs = vec![(1.0, 0.5), (0.5, 0.3)];
        let total = spreading_activation(1.0, &assocs).unwrap();
        // 1.0 + 1.0*0.5 + 0.5*0.3 = 1.65
        assert!((total - 1.65).abs() < 1e-10);
    }

    // -- Retrieval probability --

    #[test]
    fn test_retrieval_prob_above_threshold() {
        // Activation well above threshold → probability near 1.
        let p = retrieval_probability(2.0, 0.0, 0.4).unwrap();
        assert!(p > 0.99);
    }

    #[test]
    fn test_retrieval_prob_below_threshold() {
        // Activation well below threshold → probability near 0.
        let p = retrieval_probability(-2.0, 0.0, 0.4).unwrap();
        assert!(p < 0.01);
    }

    #[test]
    fn test_retrieval_prob_at_threshold() {
        // At threshold → probability = 0.5.
        let p = retrieval_probability(1.0, 1.0, 0.4).unwrap();
        assert!((p - 0.5).abs() < 1e-10);
    }

    // -- Retrieval latency --

    #[test]
    fn test_retrieval_latency_higher_activation_faster() {
        let t_high = retrieval_latency(2.0, 1.0, 1.0).unwrap();
        let t_low = retrieval_latency(0.0, 1.0, 1.0).unwrap();
        assert!(t_high < t_low);
    }

    #[test]
    fn test_retrieval_latency_known_value() {
        // A=0, F=1, f=1: T = 1 * e^0 = 1.0
        let t = retrieval_latency(0.0, 1.0, 1.0).unwrap();
        assert!((t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_retrieval_latency_reference() {
        // A=1, F=1, f=1: T = e^(-1) ≈ 0.368
        let t = retrieval_latency(1.0, 1.0, 1.0).unwrap();
        assert!((t - (-1.0_f64).exp()).abs() < 1e-10);
    }

    // -- Partial matching --

    #[test]
    fn test_partial_matching_perfect() {
        let slots = vec![(1.5, 0.0), (1.5, 0.0)];
        let mp = partial_matching(&slots).unwrap();
        assert!(mp.abs() < 1e-10); // perfect match = 0 penalty
    }

    #[test]
    fn test_partial_matching_mismatch() {
        let slots = vec![(1.5, -0.5)];
        let mp = partial_matching(&slots).unwrap();
        assert!((mp - (-0.75)).abs() < 1e-10);
    }

    #[test]
    fn test_partial_matching_invalid_similarity() {
        assert!(partial_matching(&[(1.0, 0.5)]).is_err()); // > 0
        assert!(partial_matching(&[(1.0, -1.5)]).is_err()); // < -1
    }

    // -- Serde roundtrips --

    // -- Levels of processing / Encoding --

    #[test]
    fn test_processing_level_ordering() {
        assert!(ProcessingLevel::Semantic > ProcessingLevel::Phonological);
        assert!(ProcessingLevel::Phonological > ProcessingLevel::Structural);
    }

    #[test]
    fn test_encoding_strength_deeper_better() {
        let shallow = encoding_strength(ProcessingLevel::Structural, 0.0, 1.0).unwrap();
        let deep = encoding_strength(ProcessingLevel::Semantic, 0.0, 1.0).unwrap();
        assert!(deep > shallow);
    }

    #[test]
    fn test_encoding_strength_elaboration_boosts() {
        let plain = encoding_strength(ProcessingLevel::Semantic, 0.0, 1.0).unwrap();
        let elab = encoding_strength(ProcessingLevel::Semantic, 2.0, 1.0).unwrap();
        assert!(elab > plain);
    }

    #[test]
    fn test_generation_effect_boost() {
        let read = generation_effect(0.5, false, 0.2).unwrap();
        let generated = generation_effect(0.5, true, 0.2).unwrap();
        assert!((read - 0.5).abs() < 1e-10);
        assert!(generated > read);
    }

    #[test]
    fn test_testing_effect_success() {
        let before = 0.5;
        let after = testing_effect(before, 0.3, true, 0.8).unwrap();
        assert!(after > before);
    }

    #[test]
    fn test_testing_effect_failure() {
        let before = 0.5;
        let after = testing_effect(before, 0.3, false, 0.8).unwrap();
        assert!((after - before).abs() < 1e-10);
    }

    #[test]
    fn test_testing_effect_difficulty_modulates() {
        let easy = testing_effect(0.5, 0.3, true, 0.2).unwrap();
        let hard = testing_effect(0.5, 0.3, true, 0.9).unwrap();
        assert!(hard > easy); // desirable difficulty
    }

    #[test]
    fn test_encoding_specificity_perfect_match() {
        let p = encoding_specificity(0.8, 1.0, 2.0).unwrap();
        assert!((p - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_encoding_specificity_mismatch_drops() {
        let matched = encoding_specificity(0.8, 1.0, 2.0).unwrap();
        let mismatched = encoding_specificity(0.8, 0.5, 2.0).unwrap();
        assert!(matched > mismatched);
    }

    #[test]
    fn test_processing_level_serde_roundtrip() {
        let level = ProcessingLevel::Semantic;
        let json = serde_json::to_string(&level).unwrap();
        let back: ProcessingLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(level, back);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_chunk_history_serde_roundtrip() {
        let h = ChunkHistory {
            presentation_ages: vec![1.0, 5.0, 10.0],
        };
        let json = serde_json::to_string(&h).unwrap();
        let back: ChunkHistory = serde_json::from_str(&json).unwrap();
        assert_eq!(h.presentation_ages.len(), back.presentation_ages.len());
    }
}
