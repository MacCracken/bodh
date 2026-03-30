//! Social cognition — conformity, social influence, attribution, comparison.
//!
//! Models for social psychological phenomena: Asch conformity, Latané social
//! impact, Kelley's covariation attribution, and Festinger social comparison.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite, validate_positive};

// ---------------------------------------------------------------------------
// Asch Conformity
// ---------------------------------------------------------------------------

/// Conformity pressure as a function of group characteristics (Asch, 1956).
///
/// `P(conform) = max_conformity × (1 − e^(−k × n)) × unanimity`
///
/// where `n` is the number of group members expressing the majority view,
/// `k` controls how fast conformity saturates with group size (≈ 0.3),
/// `unanimity` is the fraction of the group that is unanimous (0–1),
/// and `max_conformity` is the ceiling (Asch found ≈ 0.37 for obvious tasks).
///
/// Asch (1956) found conformity peaked at ~3–4 confederates and dropped
/// sharply when unanimity was broken.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[must_use = "returns the conformity probability without side effects"]
pub fn asch_conformity(
    group_size: usize,
    unanimity: f64,
    max_conformity: f64,
    k: f64,
) -> Result<f64> {
    validate_finite(unanimity, "unanimity")?;
    validate_finite(max_conformity, "max_conformity")?;
    validate_positive(k, "k")?;
    if !(0.0..=1.0).contains(&unanimity) {
        return Err(BodhError::InvalidParameter(
            "unanimity must be in [0, 1]".into(),
        ));
    }
    if !(0.0..=1.0).contains(&max_conformity) {
        return Err(BodhError::InvalidParameter(
            "max_conformity must be in [0, 1]".into(),
        ));
    }

    let size_effect = 1.0 - (-k * group_size as f64).exp();
    Ok(max_conformity * size_effect * unanimity)
}

// ---------------------------------------------------------------------------
// Latané's Social Impact Theory
// ---------------------------------------------------------------------------

/// Social impact magnitude (Latané, 1981).
///
/// `I = s × f(N) = s × N^t`
///
/// where `s` is the strength of each source (status, authority), `N` is
/// the number of sources, and `t` is the power exponent (typically ≈ 0.5,
/// indicating diminishing returns with each additional source).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[inline]
#[must_use = "returns the social impact without side effects"]
pub fn social_impact(source_strength: f64, num_sources: usize, exponent: f64) -> Result<f64> {
    validate_finite(source_strength, "source_strength")?;
    validate_finite(exponent, "exponent")?;
    if num_sources == 0 {
        return Ok(0.0);
    }
    Ok(source_strength * (num_sources as f64).powf(exponent))
}

/// Division of social impact among targets (Latané, 1981).
///
/// `I_per_target = total_impact / N_targets^t`
///
/// Impact is diffused when spread across multiple targets (bystander
/// effect). More targets → less impact per target.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid or
/// num_targets is zero.
#[inline]
#[must_use = "returns the per-target impact without side effects"]
pub fn social_impact_diffusion(
    total_impact: f64,
    num_targets: usize,
    exponent: f64,
) -> Result<f64> {
    validate_finite(total_impact, "total_impact")?;
    validate_finite(exponent, "exponent")?;
    if num_targets == 0 {
        return Err(BodhError::InvalidParameter(
            "num_targets must be at least 1".into(),
        ));
    }
    Ok(total_impact / (num_targets as f64).powf(exponent))
}

// ---------------------------------------------------------------------------
// Kelley's Covariation Model of Attribution
// ---------------------------------------------------------------------------

/// Information dimensions for Kelley's covariation model (1967).
///
/// Each dimension is rated on a \[0, 1\] scale where 1 = high.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CovariationInfo {
    /// Consensus: do other people respond the same way? (0 = low, 1 = high).
    pub consensus: f64,
    /// Distinctiveness: does the person respond this way to other stimuli? (0 = low/responds to many, 1 = high/only this stimulus).
    pub distinctiveness: f64,
    /// Consistency: does the person respond this way over time? (0 = low, 1 = high).
    pub consistency: f64,
}

/// Attribution type from Kelley's covariation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum AttributionType {
    /// External/stimulus attribution (the situation caused it).
    External,
    /// Internal/person attribution (the person's disposition caused it).
    Internal,
    /// Circumstantial attribution (particular circumstances caused it).
    Circumstantial,
}

/// Determine the attribution type from covariation information.
///
/// Kelley's (1967) covariation principle:
/// - **External**: high consensus, high distinctiveness, high consistency.
/// - **Internal**: low consensus, low distinctiveness, high consistency.
/// - **Circumstantial**: low consistency (regardless of other dimensions).
///
/// Uses a threshold-based classification (threshold = 0.5).
#[must_use = "returns the attribution type without side effects"]
pub fn kelley_attribution(info: &CovariationInfo) -> AttributionType {
    if info.consistency < 0.5 {
        return AttributionType::Circumstantial;
    }
    if info.consensus >= 0.5 && info.distinctiveness >= 0.5 {
        AttributionType::External
    } else {
        AttributionType::Internal
    }
}

/// Fundamental attribution error: bias toward internal attributions.
///
/// Returns a modified covariation info where consensus and distinctiveness
/// are discounted by the bias factor (0–1). A bias of 1.0 means the
/// observer completely ignores consensus/distinctiveness information,
/// always attributing internally.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if bias is outside \[0, 1\].
#[must_use = "returns the biased covariation info without side effects"]
pub fn fundamental_attribution_error(info: &CovariationInfo, bias: f64) -> Result<CovariationInfo> {
    validate_finite(bias, "bias")?;
    if !(0.0..=1.0).contains(&bias) {
        return Err(BodhError::InvalidParameter("bias must be in [0, 1]".into()));
    }
    Ok(CovariationInfo {
        consensus: info.consensus * (1.0 - bias),
        distinctiveness: info.distinctiveness * (1.0 - bias),
        consistency: info.consistency,
    })
}

// ---------------------------------------------------------------------------
// Social Comparison (Festinger, 1954)
// ---------------------------------------------------------------------------

/// Direction of social comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ComparisonDirection {
    /// Comparing with someone better (self-improvement motivation).
    Upward,
    /// Comparing with a similar other (self-evaluation).
    Lateral,
    /// Comparing with someone worse (self-enhancement).
    Downward,
}

/// Self-evaluation shift from social comparison (Festinger, 1954).
///
/// `shift = direction_weight × (other_ability − self_ability) / scale`
///
/// Positive shift = feeling better, negative = feeling worse.
/// Upward comparisons typically produce negative shifts (contrast effect),
/// downward comparisons produce positive shifts.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[inline]
#[must_use = "returns the self-evaluation shift without side effects"]
pub fn social_comparison_shift(
    self_ability: f64,
    other_ability: f64,
    direction: ComparisonDirection,
) -> Result<f64> {
    validate_finite(self_ability, "self_ability")?;
    validate_finite(other_ability, "other_ability")?;

    let raw_diff = other_ability - self_ability;

    let weight = match direction {
        ComparisonDirection::Upward => -0.7, // contrast: they're better → I feel worse
        ComparisonDirection::Lateral => -0.3, // mild evaluation effect
        ComparisonDirection::Downward => -0.6, // enhancement: they're worse → I feel better
    };

    Ok(weight * raw_diff)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Asch conformity --

    #[test]
    fn test_asch_conformity_increases_with_group() {
        let p3 = asch_conformity(3, 1.0, 0.37, 0.3).unwrap();
        let p1 = asch_conformity(1, 1.0, 0.37, 0.3).unwrap();
        assert!(p3 > p1);
    }

    #[test]
    fn test_asch_conformity_saturates() {
        let p5 = asch_conformity(5, 1.0, 0.37, 0.3).unwrap();
        let p20 = asch_conformity(20, 1.0, 0.37, 0.3).unwrap();
        // Difference should be small (saturation).
        assert!((p20 - p5).abs() < 0.1);
    }

    #[test]
    fn test_asch_conformity_broken_unanimity() {
        let unanimous = asch_conformity(5, 1.0, 0.37, 0.3).unwrap();
        let broken = asch_conformity(5, 0.5, 0.37, 0.3).unwrap();
        assert!(unanimous > broken);
    }

    #[test]
    fn test_asch_conformity_zero_group() {
        let p = asch_conformity(0, 1.0, 0.37, 0.3).unwrap();
        assert!(p.abs() < 1e-10);
    }

    #[test]
    fn test_asch_conformity_reference() {
        // Asch found ~37% conformity at peak. Our model with n=4, full
        // unanimity, max=0.37, k=0.3 should give close to 0.37.
        let p = asch_conformity(4, 1.0, 0.37, 0.3).unwrap();
        assert!(p > 0.2 && p < 0.37);
    }

    // -- Social impact --

    #[test]
    fn test_social_impact_basic() {
        // s=10, N=4, t=0.5: I = 10 * 4^0.5 = 10 * 2 = 20.
        let impact = social_impact(10.0, 4, 0.5).unwrap();
        assert!((impact - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_social_impact_zero_sources() {
        let impact = social_impact(10.0, 0, 0.5).unwrap();
        assert!(impact.abs() < 1e-10);
    }

    #[test]
    fn test_social_impact_diminishing() {
        // Adding 2 sources from a base of 1 gives more marginal gain
        // than adding 2 sources from a base of 8 (concavity of N^0.5).
        let i1 = social_impact(1.0, 1, 0.5).unwrap();
        let i3 = social_impact(1.0, 3, 0.5).unwrap();
        let i8 = social_impact(1.0, 8, 0.5).unwrap();
        let i10 = social_impact(1.0, 10, 0.5).unwrap();
        let gain_1_to_3 = i3 - i1;
        let gain_8_to_10 = i10 - i8;
        assert!(gain_1_to_3 > gain_8_to_10);
    }

    #[test]
    fn test_social_impact_diffusion() {
        let total = social_impact(10.0, 4, 0.5).unwrap();
        let per_1 = social_impact_diffusion(total, 1, 0.5).unwrap();
        let per_4 = social_impact_diffusion(total, 4, 0.5).unwrap();
        assert!(per_1 > per_4); // bystander effect
    }

    // -- Kelley attribution --

    #[test]
    fn test_kelley_external() {
        let info = CovariationInfo {
            consensus: 0.9,
            distinctiveness: 0.9,
            consistency: 0.9,
        };
        assert_eq!(kelley_attribution(&info), AttributionType::External);
    }

    #[test]
    fn test_kelley_internal() {
        let info = CovariationInfo {
            consensus: 0.1,
            distinctiveness: 0.1,
            consistency: 0.9,
        };
        assert_eq!(kelley_attribution(&info), AttributionType::Internal);
    }

    #[test]
    fn test_kelley_circumstantial() {
        let info = CovariationInfo {
            consensus: 0.9,
            distinctiveness: 0.9,
            consistency: 0.1,
        };
        assert_eq!(kelley_attribution(&info), AttributionType::Circumstantial);
    }

    #[test]
    fn test_fundamental_attribution_error_full_bias() {
        let info = CovariationInfo {
            consensus: 0.9,
            distinctiveness: 0.9,
            consistency: 0.9,
        };
        let biased = fundamental_attribution_error(&info, 1.0).unwrap();
        // Full bias → consensus and distinctiveness zeroed → internal.
        assert_eq!(kelley_attribution(&biased), AttributionType::Internal);
    }

    #[test]
    fn test_fundamental_attribution_error_no_bias() {
        let info = CovariationInfo {
            consensus: 0.9,
            distinctiveness: 0.9,
            consistency: 0.9,
        };
        let unbiased = fundamental_attribution_error(&info, 0.0).unwrap();
        assert!((unbiased.consensus - info.consensus).abs() < 1e-10);
    }

    // -- Social comparison --

    #[test]
    fn test_upward_comparison_negative_shift() {
        // Comparing with someone better → feel worse.
        let shift = social_comparison_shift(5.0, 8.0, ComparisonDirection::Upward).unwrap();
        assert!(shift < 0.0);
    }

    #[test]
    fn test_downward_comparison_positive_shift() {
        // Comparing with someone worse → feel better.
        let shift = social_comparison_shift(5.0, 3.0, ComparisonDirection::Downward).unwrap();
        assert!(shift > 0.0);
    }

    #[test]
    fn test_comparison_equal_ability() {
        let shift = social_comparison_shift(5.0, 5.0, ComparisonDirection::Lateral).unwrap();
        assert!(shift.abs() < 1e-10);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_covariation_info_serde_roundtrip() {
        let info = CovariationInfo {
            consensus: 0.8,
            distinctiveness: 0.3,
            consistency: 0.9,
        };
        let json = serde_json::to_string(&info).unwrap();
        let back: CovariationInfo = serde_json::from_str(&json).unwrap();
        assert!((info.consensus - back.consensus).abs() < 1e-10);
    }

    #[test]
    fn test_attribution_type_serde_roundtrip() {
        let a = AttributionType::External;
        let json = serde_json::to_string(&a).unwrap();
        let back: AttributionType = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn test_comparison_direction_serde_roundtrip() {
        let d = ComparisonDirection::Upward;
        let json = serde_json::to_string(&d).unwrap();
        let back: ComparisonDirection = serde_json::from_str(&json).unwrap();
        assert_eq!(d, back);
    }
}
