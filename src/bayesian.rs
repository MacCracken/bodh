//! Bayesian inference — belief updating, base rate neglect, conservatism.
//!
//! Models for rational and biased probability updating based on evidence.

use crate::error::{BodhError, Result, validate_finite};

// ---------------------------------------------------------------------------
// Bayes' theorem
// ---------------------------------------------------------------------------

/// Bayesian posterior probability.
///
/// `P(H|E) = P(E|H) × P(H) / P(E)`
///
/// where `P(E) = P(E|H) × P(H) + P(E|¬H) × (1 − P(H))`.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any probability is outside \[0, 1\]
/// or if `P(E)` is zero (impossible evidence).
#[must_use = "returns the posterior probability without side effects"]
pub fn bayes_posterior(prior: f64, likelihood: f64, likelihood_complement: f64) -> Result<f64> {
    validate_probability(prior, "prior")?;
    validate_probability(likelihood, "likelihood")?;
    validate_probability(likelihood_complement, "likelihood_complement")?;

    let p_evidence = likelihood * prior + likelihood_complement * (1.0 - prior);
    if p_evidence.abs() < f64::EPSILON {
        return Err(BodhError::ComputationError(
            "P(E) is zero — impossible evidence".into(),
        ));
    }
    Ok((likelihood * prior) / p_evidence)
}

/// Likelihood ratio (Bayes factor).
///
/// `LR = P(E|H) / P(E|¬H)`
///
/// LR > 1 means evidence supports H, LR < 1 means evidence supports ¬H.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if probabilities are invalid or
/// `likelihood_complement` is zero.
#[inline]
#[must_use = "returns the likelihood ratio without side effects"]
pub fn likelihood_ratio(likelihood: f64, likelihood_complement: f64) -> Result<f64> {
    validate_probability(likelihood, "likelihood")?;
    if likelihood_complement <= 0.0
        || likelihood_complement > 1.0
        || !likelihood_complement.is_finite()
    {
        return Err(BodhError::InvalidParameter(
            "likelihood_complement must be in (0, 1]".into(),
        ));
    }
    Ok(likelihood / likelihood_complement)
}

/// Convert prior odds + likelihood ratio to posterior odds.
///
/// `posterior_odds = prior_odds × LR`
///
/// This is the odds form of Bayes' theorem.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or negative.
#[inline]
#[must_use = "returns posterior odds without side effects"]
pub fn posterior_odds(prior_odds: f64, lr: f64) -> Result<f64> {
    validate_finite(prior_odds, "prior_odds")?;
    validate_finite(lr, "lr")?;
    if prior_odds < 0.0 {
        return Err(BodhError::InvalidParameter(
            "prior_odds must be non-negative".into(),
        ));
    }
    if lr < 0.0 {
        return Err(BodhError::InvalidParameter(
            "likelihood_ratio must be non-negative".into(),
        ));
    }
    Ok(prior_odds * lr)
}

/// Convert odds to probability.
///
/// `P = odds / (1 + odds)`
#[inline]
#[must_use]
pub fn odds_to_probability(odds: f64) -> f64 {
    if odds.is_infinite() && odds > 0.0 {
        return 1.0;
    }
    odds / (1.0 + odds)
}

/// Convert probability to odds.
///
/// `odds = P / (1 − P)`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if p is outside \[0, 1) or non-finite.
#[inline]
#[must_use = "returns odds without side effects"]
pub fn probability_to_odds(p: f64) -> Result<f64> {
    validate_finite(p, "p")?;
    if !(0.0..1.0).contains(&p) {
        return Err(BodhError::InvalidParameter(
            "probability must be in [0, 1)".into(),
        ));
    }
    Ok(p / (1.0 - p))
}

// ---------------------------------------------------------------------------
// Cognitive biases in Bayesian reasoning
// ---------------------------------------------------------------------------

/// Base rate neglect: how humans underweight prior probability.
///
/// Models the common bias where people rely too heavily on the
/// likelihood (diagnostic evidence) and ignore the prior (base rate).
///
/// `biased_posterior = bayes(prior_eff, likelihood, likelihood_comp)`
///
/// where `prior_eff = base_rate_weight × prior + (1 − base_rate_weight) × 0.5`
///
/// `base_rate_weight = 0.0` means complete base rate neglect (prior = 0.5).
/// `base_rate_weight = 1.0` means rational Bayesian.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if probabilities are invalid or
/// `base_rate_weight` is outside \[0, 1\].
#[must_use = "returns the biased posterior without side effects"]
pub fn base_rate_neglect(
    prior: f64,
    likelihood: f64,
    likelihood_complement: f64,
    base_rate_weight: f64,
) -> Result<f64> {
    validate_probability(prior, "prior")?;
    validate_probability(likelihood, "likelihood")?;
    validate_probability(likelihood_complement, "likelihood_complement")?;
    validate_finite(base_rate_weight, "base_rate_weight")?;
    if !(0.0..=1.0).contains(&base_rate_weight) {
        return Err(BodhError::InvalidParameter(
            "base_rate_weight must be in [0, 1]".into(),
        ));
    }

    let effective_prior = base_rate_weight * prior + (1.0 - base_rate_weight) * 0.5;
    bayes_posterior(effective_prior, likelihood, likelihood_complement)
}

/// Conservative Bayesian updating (Edwards, 1968).
///
/// Humans tend to update beliefs less than Bayes' theorem prescribes.
/// The conservatism factor `c` in \(0, 1\] controls how much of the
/// Bayesian update is actually applied.
///
/// `updated = prior + c × (bayesian_posterior − prior)`
///
/// `c = 1.0` is fully rational, `c = 0.0` means no updating at all.
/// Edwards (1968) found typical c ≈ 0.3–0.7 in experiments.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if probabilities or `c` are invalid.
#[must_use = "returns the conservatively updated belief without side effects"]
pub fn conservative_updating(
    prior: f64,
    likelihood: f64,
    likelihood_complement: f64,
    conservatism: f64,
) -> Result<f64> {
    validate_probability(prior, "prior")?;
    validate_finite(conservatism, "conservatism")?;
    if !(0.0..=1.0).contains(&conservatism) {
        return Err(BodhError::InvalidParameter(
            "conservatism must be in [0, 1]".into(),
        ));
    }
    let bayesian = bayes_posterior(prior, likelihood, likelihood_complement)?;
    Ok(prior + conservatism * (bayesian - prior))
}

/// Sequential Bayesian updating: update a belief through multiple
/// pieces of evidence.
///
/// Applies Bayes' theorem iteratively, using the posterior from each
/// step as the prior for the next.
///
/// Each evidence item is a `(likelihood, likelihood_complement)` pair.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any probability is invalid
/// or if `P(E)` becomes zero at any step.
#[must_use = "returns the final posterior without side effects"]
pub fn sequential_update(prior: f64, evidence: &[(f64, f64)]) -> Result<f64> {
    let mut current = prior;
    for (i, &(lh, lh_comp)) in evidence.iter().enumerate() {
        current = bayes_posterior(current, lh, lh_comp)
            .map_err(|e| BodhError::ComputationError(format!("at evidence[{i}]: {e}")))?;
    }
    Ok(current)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate that a value is a valid probability in \[0, 1\].
#[inline]
fn validate_probability(p: f64, name: &str) -> Result<()> {
    validate_finite(p, name)?;
    if !(0.0..=1.0).contains(&p) {
        return Err(BodhError::InvalidParameter(format!(
            "{name} must be in [0, 1], got {p}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Bayes' theorem --

    #[test]
    fn test_bayes_classic_medical() {
        // Classic: disease prevalence 1%, test sensitivity 99%, specificity 95%.
        // P(H|E) = 0.99 * 0.01 / (0.99*0.01 + 0.05*0.99) = 0.0099 / 0.0594 ≈ 0.167.
        let posterior = bayes_posterior(0.01, 0.99, 0.05).unwrap();
        assert!((posterior - 0.1667).abs() < 0.01);
    }

    #[test]
    fn test_bayes_certain_evidence() {
        // P(E|H) = 1.0, P(E|¬H) = 0.0 → posterior = 1.0 (if prior > 0).
        // But P(E|¬H) = 0 makes P(E) = prior, so posterior = 1.0.
        let posterior = bayes_posterior(0.5, 1.0, 0.0).unwrap();
        assert!((posterior - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_uninformative_evidence() {
        // P(E|H) = P(E|¬H) → posterior = prior.
        let posterior = bayes_posterior(0.3, 0.5, 0.5).unwrap();
        assert!((posterior - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_bayes_prior_zero() {
        // Prior = 0 → posterior = 0 regardless of evidence.
        let posterior = bayes_posterior(0.0, 0.99, 0.05).unwrap();
        assert!(posterior.abs() < 1e-10);
    }

    #[test]
    fn test_bayes_invalid() {
        assert!(bayes_posterior(-0.1, 0.5, 0.5).is_err());
        assert!(bayes_posterior(0.5, 1.5, 0.5).is_err());
    }

    // -- Likelihood ratio --

    #[test]
    fn test_likelihood_ratio_basic() {
        let lr = likelihood_ratio(0.9, 0.1).unwrap();
        assert!((lr - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_likelihood_ratio_neutral() {
        let lr = likelihood_ratio(0.5, 0.5).unwrap();
        assert!((lr - 1.0).abs() < 1e-10);
    }

    // -- Odds conversion --

    #[test]
    fn test_odds_probability_roundtrip() {
        let p = 0.75;
        let odds = probability_to_odds(p).unwrap();
        assert!((odds - 3.0).abs() < 1e-10);
        let back = odds_to_probability(odds);
        assert!((back - p).abs() < 1e-10);
    }

    #[test]
    fn test_odds_to_probability_zero() {
        assert!(odds_to_probability(0.0).abs() < 1e-10);
    }

    #[test]
    fn test_odds_to_probability_infinity() {
        assert!((odds_to_probability(f64::INFINITY) - 1.0).abs() < 1e-10);
    }

    // -- Posterior odds --

    #[test]
    fn test_posterior_odds_basic() {
        // Prior odds 1:1 (p=0.5), LR=9 → posterior odds = 9:1.
        let po = posterior_odds(1.0, 9.0).unwrap();
        assert!((po - 9.0).abs() < 1e-10);
        let p = odds_to_probability(po);
        assert!((p - 0.9).abs() < 1e-10);
    }

    // -- Base rate neglect --

    #[test]
    fn test_base_rate_neglect_full() {
        // Full neglect (weight=0) → treats prior as 0.5.
        let biased = base_rate_neglect(0.01, 0.99, 0.05, 0.0).unwrap();
        let rational = bayes_posterior(0.01, 0.99, 0.05).unwrap();
        assert!(biased > rational); // neglecting low base rate inflates posterior
    }

    #[test]
    fn test_base_rate_neglect_rational() {
        // Weight = 1.0 → same as rational Bayesian.
        let biased = base_rate_neglect(0.3, 0.8, 0.2, 1.0).unwrap();
        let rational = bayes_posterior(0.3, 0.8, 0.2).unwrap();
        assert!((biased - rational).abs() < 1e-10);
    }

    // -- Conservative updating --

    #[test]
    fn test_conservative_updating_full() {
        // c=1.0 → fully rational.
        let updated = conservative_updating(0.3, 0.9, 0.1, 1.0).unwrap();
        let rational = bayes_posterior(0.3, 0.9, 0.1).unwrap();
        assert!((updated - rational).abs() < 1e-10);
    }

    #[test]
    fn test_conservative_updating_none() {
        // c=0.0 → no update.
        let updated = conservative_updating(0.3, 0.9, 0.1, 0.0).unwrap();
        assert!((updated - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_conservative_updating_partial() {
        // c=0.5 → halfway between prior and posterior.
        let prior = 0.3;
        let rational = bayes_posterior(prior, 0.9, 0.1).unwrap();
        let updated = conservative_updating(prior, 0.9, 0.1, 0.5).unwrap();
        let expected = prior + 0.5 * (rational - prior);
        assert!((updated - expected).abs() < 1e-10);
    }

    // -- Sequential update --

    #[test]
    fn test_sequential_update_single() {
        let posterior = sequential_update(0.5, &[(0.9, 0.1)]).unwrap();
        let direct = bayes_posterior(0.5, 0.9, 0.1).unwrap();
        assert!((posterior - direct).abs() < 1e-10);
    }

    #[test]
    fn test_sequential_update_multiple() {
        // Two pieces of confirming evidence should increase belief.
        let posterior = sequential_update(0.5, &[(0.8, 0.2), (0.8, 0.2)]).unwrap();
        let single = sequential_update(0.5, &[(0.8, 0.2)]).unwrap();
        assert!(posterior > single);
    }

    #[test]
    fn test_sequential_update_empty() {
        // No evidence → posterior = prior.
        let posterior = sequential_update(0.7, &[]).unwrap();
        assert!((posterior - 0.7).abs() < 1e-10);
    }
}
