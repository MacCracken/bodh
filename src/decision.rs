//! Decision-making — prospect theory, expected utility, bounded rationality.

use crate::error::{BodhError, Result, validate_finite, validate_positive};

/// Prospect theory value function (Kahneman & Tversky, 1979).
///
/// For gains (`outcome >= reference`):
///   `v = (outcome - reference)^alpha`
///
/// For losses (`outcome < reference`):
///   `v = -lambda * (reference - outcome)^beta`
///
/// Typical parameters: alpha = 0.88, beta = 0.88, lambda = 2.25.
///
/// The function is concave for gains (risk aversion) and convex for
/// losses (risk seeking), with losses weighted more heavily (loss aversion).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if alpha, beta, or lambda are non-positive.
#[must_use = "returns the subjective value without side effects"]
pub fn prospect_theory_value(
    outcome: f64,
    reference: f64,
    alpha: f64,
    beta: f64,
    lambda: f64,
) -> Result<f64> {
    validate_finite(outcome, "outcome")?;
    validate_finite(reference, "reference")?;
    validate_positive(alpha, "alpha")?;
    validate_positive(beta, "beta")?;
    validate_positive(lambda, "lambda")?;

    let x = outcome - reference;
    if x >= 0.0 {
        Ok(x.powf(alpha))
    } else {
        Ok(-lambda * (-x).powf(beta))
    }
}

/// Expected utility: weighted sum of utilities.
///
/// `EU = sum(p_i * u_i)` for each (probability, utility) pair.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any probability or utility is non-finite.
#[must_use = "returns the expected utility without side effects"]
pub fn expected_utility(outcomes: &[(f64, f64)]) -> Result<f64> {
    let mut eu = 0.0;
    for (prob, utility) in outcomes {
        validate_finite(*prob, "probability")?;
        validate_finite(*utility, "utility")?;
        eu += prob * utility;
    }
    Ok(eu)
}

/// Anchoring bias model: adjusted estimate from an anchor.
///
/// `estimate = anchor + adjustment_factor * (true_value - anchor)`
///
/// When `adjustment_factor = 1.0`, the estimate equals the true value.
/// When `adjustment_factor = 0.0`, the estimate equals the anchor
/// (complete anchoring).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if parameters are non-finite or
/// adjustment_factor is outside [0, 1].
#[inline]
#[must_use = "returns the biased estimate without side effects"]
pub fn anchoring_bias(anchor: f64, true_value: f64, adjustment_factor: f64) -> Result<f64> {
    validate_finite(anchor, "anchor")?;
    validate_finite(true_value, "true_value")?;
    validate_finite(adjustment_factor, "adjustment_factor")?;
    if !(0.0..=1.0).contains(&adjustment_factor) {
        return Err(BodhError::InvalidParameter(
            "adjustment_factor must be in [0, 1]".into(),
        ));
    }
    Ok(anchor + adjustment_factor * (true_value - anchor))
}

/// Satisficing model (Simon's bounded rationality).
///
/// Given a list of options with utility values, returns the index of the
/// first option that meets or exceeds the aspiration level. Returns `None`
/// if no option is satisfactory.
#[must_use = "returns the satisficing choice without side effects"]
pub fn satisfice(options: &[f64], aspiration_level: f64) -> Option<usize> {
    options
        .iter()
        .position(|&utility| utility >= aspiration_level)
}

/// Regret for a choice: difference between the best possible outcome
/// and the chosen outcome.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[inline]
#[must_use = "returns the regret value without side effects"]
pub fn regret(chosen_outcome: f64, best_outcome: f64) -> Result<f64> {
    validate_finite(chosen_outcome, "chosen_outcome")?;
    validate_finite(best_outcome, "best_outcome")?;
    Ok((best_outcome - chosen_outcome).max(0.0))
}

/// Probability weighting function (Tversky & Kahneman, 1992).
///
/// `w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)`
///
/// Overweights small probabilities, underweights moderate-to-high ones.
/// Typical gamma ≈ 0.61.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if p is outside [0, 1] or gamma is non-positive.
#[inline]
#[must_use = "returns the decision weight without side effects"]
pub fn probability_weighting(p: f64, gamma: f64) -> Result<f64> {
    validate_finite(p, "p")?;
    validate_positive(gamma, "gamma")?;
    if !(0.0..=1.0).contains(&p) {
        return Err(BodhError::InvalidParameter(
            "probability must be in [0, 1]".into(),
        ));
    }
    if p == 0.0 {
        return Ok(0.0);
    }
    if (p - 1.0).abs() < f64::EPSILON {
        return Ok(1.0);
    }
    let pg = p.powf(gamma);
    let qg = (1.0 - p).powf(gamma);
    Ok(pg / (pg + qg).powf(1.0 / gamma))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prospect_theory_gain() {
        // Gain of 100 with alpha=0.88, lambda=2.25: v = 100^0.88.
        let v = prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        let expected = 100.0_f64.powf(0.88);
        assert!((v - expected).abs() < 1e-6);
    }

    #[test]
    fn test_prospect_theory_loss() {
        // Loss of 100: v = -2.25 * 100^0.88.
        let v = prospect_theory_value(0.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        let expected = -2.25 * 100.0_f64.powf(0.88);
        assert!((v - expected).abs() < 1e-6);
    }

    #[test]
    fn test_prospect_theory_reference_point() {
        // At reference point, value is 0.
        let v = prospect_theory_value(100.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        assert!((v - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_prospect_theory_loss_aversion() {
        // Loss of X hurts more than gain of X feels good (lambda > 1).
        let gain = prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        let loss = prospect_theory_value(0.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        assert!(gain.abs() < loss.abs()); // loss hurts more
    }

    #[test]
    fn test_prospect_theory_concave_gains() {
        // For gains, value function should be concave (diminishing sensitivity).
        let v50 = prospect_theory_value(150.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        let v100 = prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
        // v(100) should be less than 2 * v(50) if concave.
        assert!(v100 < 2.0 * v50);
    }

    #[test]
    fn test_expected_utility() {
        let outcomes = vec![(0.5, 100.0), (0.5, 0.0)];
        let eu = expected_utility(&outcomes).unwrap();
        assert!((eu - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_expected_utility_certain() {
        let outcomes = vec![(1.0, 42.0)];
        let eu = expected_utility(&outcomes).unwrap();
        assert!((eu - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_anchoring_bias_no_adjustment() {
        let est = anchoring_bias(50.0, 100.0, 0.0).unwrap();
        assert!((est - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_anchoring_bias_full_adjustment() {
        let est = anchoring_bias(50.0, 100.0, 1.0).unwrap();
        assert!((est - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_anchoring_bias_partial() {
        let est = anchoring_bias(50.0, 100.0, 0.5).unwrap();
        assert!((est - 75.0).abs() < 1e-10);
    }

    #[test]
    fn test_satisfice_found() {
        let options = vec![3.0, 5.0, 8.0, 2.0];
        assert_eq!(satisfice(&options, 5.0), Some(1));
    }

    #[test]
    fn test_satisfice_none() {
        let options = vec![1.0, 2.0, 3.0];
        assert_eq!(satisfice(&options, 10.0), None);
    }

    #[test]
    fn test_regret() {
        let r = regret(70.0, 100.0).unwrap();
        assert!((r - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_regret_no_regret() {
        let r = regret(100.0, 80.0).unwrap();
        assert!((r - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_probability_weighting_extremes() {
        let w0 = probability_weighting(0.0, 0.61).unwrap();
        let w1 = probability_weighting(1.0, 0.61).unwrap();
        assert!((w0 - 0.0).abs() < 1e-10);
        assert!((w1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_probability_weighting_overweights_small() {
        // Small probabilities should be overweighted: w(0.05) > 0.05.
        let w = probability_weighting(0.05, 0.61).unwrap();
        assert!(w > 0.05);
    }

    #[test]
    fn test_probability_weighting_underweights_moderate() {
        // Moderate-to-high probabilities should be underweighted: w(0.9) < 0.9.
        let w = probability_weighting(0.9, 0.61).unwrap();
        assert!(w < 0.9);
    }

    #[test]
    fn test_probability_weighting_invalid() {
        assert!(probability_weighting(-0.1, 0.61).is_err());
        assert!(probability_weighting(1.1, 0.61).is_err());
        assert!(probability_weighting(0.5, 0.0).is_err());
    }
}
