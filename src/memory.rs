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
