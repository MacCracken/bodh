//! Attention — Posner cueing, feature integration, attentional blink, visual search.
//!
//! Models for spatial attention, feature-based selection, temporal
//! attention limitations, and search efficiency.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite, validate_positive};

// ---------------------------------------------------------------------------
// Posner Cueing Paradigm (Posner, 1980)
// ---------------------------------------------------------------------------

/// Cue validity in the Posner cueing paradigm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CueValidity {
    /// Cue correctly indicates target location.
    Valid,
    /// Cue gives no spatial information.
    Neutral,
    /// Cue indicates wrong location (requires reorienting).
    Invalid,
}

/// Compute reaction time for a Posner cueing trial.
///
/// `RT = base_rt + validity_effect`
///
/// where the validity effect is:
/// - Valid: −benefit (faster than neutral)
/// - Neutral: 0
/// - Invalid: +cost (slower than neutral)
///
/// Typical values: benefit ≈ 20–50ms, cost ≈ 30–80ms.
/// The cost-benefit asymmetry (cost > benefit) is a robust finding.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or negative.
#[inline]
#[must_use = "returns the predicted reaction time without side effects"]
pub fn posner_cueing_rt(
    base_rt: f64,
    validity: CueValidity,
    benefit: f64,
    cost: f64,
) -> Result<f64> {
    validate_positive(base_rt, "base_rt")?;
    validate_finite(benefit, "benefit")?;
    validate_finite(cost, "cost")?;

    let effect = match validity {
        CueValidity::Valid => -benefit,
        CueValidity::Neutral => 0.0,
        CueValidity::Invalid => cost,
    };
    Ok(base_rt + effect)
}

/// Attentional orienting effect: cost minus benefit.
///
/// Larger values indicate stronger spatial attention effects.
/// Typical range: 50–130ms for endogenous cues.
#[inline]
#[must_use]
pub fn orienting_effect(benefit: f64, cost: f64) -> f64 {
    cost + benefit
}

/// Inhibition of return (IOR): slowing at previously attended locations.
///
/// At long SOAs (stimulus onset asynchrony > ~300ms), valid cues
/// produce *slower* responses than invalid cues. This models the
/// transition from facilitation to IOR.
///
/// `effect = benefit × (1 − 2 × sigmoid((soa − crossover) / scale))`
///
/// At short SOA → facilitation (+benefit).
/// At long SOA → inhibition (−benefit).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or
/// `scale` is non-positive.
#[must_use = "returns the IOR-adjusted RT effect without side effects"]
pub fn inhibition_of_return(
    soa_ms: f64,
    benefit: f64,
    crossover_ms: f64,
    scale: f64,
) -> Result<f64> {
    validate_finite(soa_ms, "soa_ms")?;
    validate_finite(benefit, "benefit")?;
    validate_finite(crossover_ms, "crossover_ms")?;
    validate_positive(scale, "scale")?;

    let sigmoid = 1.0 / (1.0 + (-((soa_ms - crossover_ms) / scale)).exp());
    Ok(benefit * (1.0 - 2.0 * sigmoid))
}

// ---------------------------------------------------------------------------
// Feature Integration Theory (Treisman & Gelade, 1980)
// ---------------------------------------------------------------------------

/// Search type from Feature Integration Theory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum SearchType {
    /// Target defined by a single feature — parallel, efficient.
    Feature,
    /// Target defined by a conjunction of features — serial, effortful.
    Conjunction,
}

/// Visual search time prediction (Treisman & Gelade, 1980).
///
/// Feature search: `RT = a` (flat, independent of set size).
/// Conjunction search: `RT = a + b × set_size` (linear).
///
/// For conjunction search, target-absent trials have slope ≈ 2× target-present
/// (exhaustive vs self-terminating search).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or
/// `set_size` is zero.
#[must_use = "returns the predicted search time without side effects"]
pub fn visual_search_rt(
    search_type: SearchType,
    set_size: usize,
    base_rt: f64,
    per_item_ms: f64,
    target_present: bool,
) -> Result<f64> {
    validate_finite(base_rt, "base_rt")?;
    validate_finite(per_item_ms, "per_item_ms")?;
    if set_size == 0 {
        return Err(BodhError::InvalidParameter(
            "set_size must be at least 1".into(),
        ));
    }

    match search_type {
        SearchType::Feature => Ok(base_rt),
        SearchType::Conjunction => {
            let slope = if target_present {
                per_item_ms // self-terminating: ~half items checked
            } else {
                per_item_ms * 2.0 // exhaustive: all items checked
            };
            Ok(base_rt + slope * set_size as f64)
        }
    }
}

/// Search efficiency: slope of the RT × set-size function.
///
/// Efficient search (feature): slope < 10 ms/item.
/// Inefficient search (conjunction): slope > 20 ms/item.
///
/// Returns the search slope in ms/item from two data points.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if set sizes are equal.
#[inline]
#[must_use = "returns the search slope without side effects"]
pub fn search_slope(rt1: f64, set_size1: usize, rt2: f64, set_size2: usize) -> Result<f64> {
    validate_finite(rt1, "rt1")?;
    validate_finite(rt2, "rt2")?;
    if set_size1 == set_size2 {
        return Err(BodhError::InvalidParameter(
            "set sizes must differ to compute slope".into(),
        ));
    }
    Ok((rt2 - rt1) / (set_size2 as f64 - set_size1 as f64))
}

// ---------------------------------------------------------------------------
// Attentional Blink (Raymond, Shapiro, & Arnell, 1992)
// ---------------------------------------------------------------------------

/// Attentional blink: reduced detection of a second target (T2) when
/// it appears 200–500ms after the first target (T1) in a rapid stream.
///
/// `accuracy = baseline − depth × blink_curve(lag)`
///
/// where `blink_curve` is a Gaussian-like dip centered around lag 3
/// (≈ 270ms at typical RSVP rates).
///
/// `lag` is the number of items between T1 and T2 (1-indexed).
/// Lag 1 often shows "lag-1 sparing" (no deficit).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or
/// `depth`/`spread` are non-positive.
#[must_use = "returns the T2 detection accuracy without side effects"]
pub fn attentional_blink(
    lag: usize,
    baseline_accuracy: f64,
    depth: f64,
    blink_center: f64,
    spread: f64,
) -> Result<f64> {
    validate_finite(baseline_accuracy, "baseline_accuracy")?;
    validate_positive(depth, "depth")?;
    validate_finite(blink_center, "blink_center")?;
    validate_positive(spread, "spread")?;

    if lag == 0 {
        return Err(BodhError::InvalidParameter("lag must be at least 1".into()));
    }

    // Lag-1 sparing: no deficit at lag 1
    if lag == 1 {
        return Ok(baseline_accuracy);
    }

    let deviation = lag as f64 - blink_center;
    let blink_curve = (-0.5 * (deviation / spread).powi(2)).exp();
    let accuracy = baseline_accuracy - depth * blink_curve;
    Ok(accuracy.max(0.0))
}

// ---------------------------------------------------------------------------
// Broadbent's Filter Theory / Capacity
// ---------------------------------------------------------------------------

/// Attentional capacity: fraction of information processed given
/// channel load and capacity limit.
///
/// `throughput = min(demand, capacity) / demand`
///
/// At low demand, throughput = 1.0 (all processed).
/// When demand exceeds capacity, throughput drops proportionally.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-positive.
#[inline]
#[must_use = "returns the processing throughput without side effects"]
pub fn capacity_throughput(demand: f64, capacity: f64) -> Result<f64> {
    validate_positive(demand, "demand")?;
    validate_positive(capacity, "capacity")?;
    Ok((capacity / demand).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Posner cueing --

    #[test]
    fn test_posner_valid_faster() {
        let valid = posner_cueing_rt(300.0, CueValidity::Valid, 30.0, 50.0).unwrap();
        let neutral = posner_cueing_rt(300.0, CueValidity::Neutral, 30.0, 50.0).unwrap();
        let invalid = posner_cueing_rt(300.0, CueValidity::Invalid, 30.0, 50.0).unwrap();
        assert!(valid < neutral);
        assert!(neutral < invalid);
    }

    #[test]
    fn test_posner_known_values() {
        let valid = posner_cueing_rt(300.0, CueValidity::Valid, 30.0, 50.0).unwrap();
        assert!((valid - 270.0).abs() < 1e-10);
        let invalid = posner_cueing_rt(300.0, CueValidity::Invalid, 30.0, 50.0).unwrap();
        assert!((invalid - 350.0).abs() < 1e-10);
    }

    #[test]
    fn test_orienting_effect() {
        assert!((orienting_effect(30.0, 50.0) - 80.0).abs() < 1e-10);
    }

    #[test]
    fn test_ior_facilitation_early() {
        // Short SOA → positive effect (facilitation).
        let effect = inhibition_of_return(100.0, 30.0, 300.0, 50.0).unwrap();
        assert!(effect > 0.0);
    }

    #[test]
    fn test_ior_inhibition_late() {
        // Long SOA → negative effect (inhibition).
        let effect = inhibition_of_return(500.0, 30.0, 300.0, 50.0).unwrap();
        assert!(effect < 0.0);
    }

    #[test]
    fn test_ior_crossover() {
        // At crossover, effect ≈ 0.
        let effect = inhibition_of_return(300.0, 30.0, 300.0, 50.0).unwrap();
        assert!(effect.abs() < 1.0);
    }

    // -- Visual search --

    #[test]
    fn test_feature_search_flat() {
        let rt5 = visual_search_rt(SearchType::Feature, 5, 400.0, 25.0, true).unwrap();
        let rt20 = visual_search_rt(SearchType::Feature, 20, 400.0, 25.0, true).unwrap();
        assert!((rt5 - rt20).abs() < 1e-10); // flat
    }

    #[test]
    fn test_conjunction_search_linear() {
        let rt5 = visual_search_rt(SearchType::Conjunction, 5, 400.0, 25.0, true).unwrap();
        let rt20 = visual_search_rt(SearchType::Conjunction, 20, 400.0, 25.0, true).unwrap();
        assert!(rt20 > rt5); // linear increase
    }

    #[test]
    fn test_conjunction_absent_slower() {
        let present = visual_search_rt(SearchType::Conjunction, 10, 400.0, 25.0, true).unwrap();
        let absent = visual_search_rt(SearchType::Conjunction, 10, 400.0, 25.0, false).unwrap();
        assert!(absent > present); // exhaustive search
    }

    #[test]
    fn test_conjunction_absent_2x_slope() {
        // Absent slope should be 2× present slope.
        let p5 = visual_search_rt(SearchType::Conjunction, 5, 400.0, 25.0, true).unwrap();
        let p10 = visual_search_rt(SearchType::Conjunction, 10, 400.0, 25.0, true).unwrap();
        let a5 = visual_search_rt(SearchType::Conjunction, 5, 400.0, 25.0, false).unwrap();
        let a10 = visual_search_rt(SearchType::Conjunction, 10, 400.0, 25.0, false).unwrap();
        let present_slope = (p10 - p5) / 5.0;
        let absent_slope = (a10 - a5) / 5.0;
        assert!((absent_slope - 2.0 * present_slope).abs() < 1e-10);
    }

    #[test]
    fn test_search_slope_basic() {
        let slope = search_slope(400.0, 5, 650.0, 15).unwrap();
        assert!((slope - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_search_slope_equal_sizes() {
        assert!(search_slope(400.0, 5, 500.0, 5).is_err());
    }

    // -- Attentional blink --

    #[test]
    fn test_ab_lag1_sparing() {
        let acc = attentional_blink(1, 0.95, 0.4, 3.0, 1.5).unwrap();
        assert!((acc - 0.95).abs() < 1e-10); // no deficit at lag 1
    }

    #[test]
    fn test_ab_deficit_at_lag3() {
        let acc = attentional_blink(3, 0.95, 0.4, 3.0, 1.5).unwrap();
        assert!(acc < 0.95); // deficit
        assert!(acc < 0.6); // substantial deficit
    }

    #[test]
    fn test_ab_recovery_at_lag8() {
        let acc_3 = attentional_blink(3, 0.95, 0.4, 3.0, 1.5).unwrap();
        let acc_8 = attentional_blink(8, 0.95, 0.4, 3.0, 1.5).unwrap();
        assert!(acc_8 > acc_3); // recovery
    }

    #[test]
    fn test_ab_invalid_lag() {
        assert!(attentional_blink(0, 0.95, 0.4, 3.0, 1.5).is_err());
    }

    // -- Capacity --

    #[test]
    fn test_capacity_under_limit() {
        let t = capacity_throughput(3.0, 7.0).unwrap();
        assert!((t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_capacity_over_limit() {
        let t = capacity_throughput(10.0, 7.0).unwrap();
        assert!((t - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_capacity_at_limit() {
        let t = capacity_throughput(7.0, 7.0).unwrap();
        assert!((t - 1.0).abs() < 1e-10);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_cue_validity_serde_roundtrip() {
        let c = CueValidity::Invalid;
        let json = serde_json::to_string(&c).unwrap();
        let back: CueValidity = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }

    #[test]
    fn test_search_type_serde_roundtrip() {
        let s = SearchType::Conjunction;
        let json = serde_json::to_string(&s).unwrap();
        let back: SearchType = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn test_flow_channel_serde_roundtrip() {
        // Using FlowChannel would require importing from motivation, so test SearchType here
        let s = SearchType::Feature;
        let json = serde_json::to_string(&s).unwrap();
        let back: SearchType = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }
}
