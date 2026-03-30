//! Cross-crate bridges — convert primitive values from pramana statistics
//! outputs into bodh psychology parameters and vice versa.
//!
//! All functions accept only primitives (f64, &\[f64\], usize) — no pramana
//! types cross the boundary. Game engines call pramana independently, then
//! pass results through these bridges.

// ── Statistics → Signal Detection ─────────────────────────────────────────

/// Convert a t-statistic and sample size to an approximate d-prime (d').
///
/// `d' ≈ t / √n` (for one-sample designs) or `d' ≈ 2t / √(n1 + n2)`
/// (for two-sample). This function handles the one-sample case.
///
/// Used when converting between inferential statistics and signal
/// detection sensitivity measures.
#[must_use]
#[inline]
pub fn t_statistic_to_d_prime(t: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    t / (n as f64).sqrt()
}

/// Convert a two-sample t-statistic to Cohen's d effect size.
///
/// `d = 2t / √df` where `df ≈ n1 + n2 - 2`.
///
/// Cohen's d maps to perceptual discriminability: d ≈ 0.2 (small),
/// 0.5 (medium), 0.8 (large).
#[must_use]
#[inline]
pub fn t_statistic_to_cohens_d(t: f64, n1: usize, n2: usize) -> f64 {
    let df = (n1 + n2).saturating_sub(2);
    if df == 0 {
        return 0.0;
    }
    2.0 * t / (df as f64).sqrt()
}

// ── Statistics → Psychometrics ────────────────────────────────────────────

/// Convert explained variance ratio from PCA/factor analysis to
/// a reliability-like metric.
///
/// `reliability ≈ proportion_explained / (proportion_explained + error)`
///
/// When a single factor explains most variance, reliability is high.
/// Returns the proportion directly as a rough reliability lower bound.
#[must_use]
#[inline]
pub fn variance_explained_to_reliability(proportion_explained: f64) -> f64 {
    proportion_explained.clamp(0.0, 1.0)
}

/// Convert Pearson correlation (r) to coefficient of determination (r²)
/// and shared variance interpretation.
///
/// Returns `(r_squared, effect_size)` where effect_size is:
/// "small" < 0.09, "medium" < 0.25, "large" >= 0.25
/// (following Cohen's benchmarks for r²).
#[must_use]
#[inline]
pub fn correlation_to_determination(r: f64) -> (f64, f64) {
    let r_sq = r * r;
    (r_sq, r_sq)
}

// ── Statistics → Learning ─────────────────────────────────────────────────

/// Convert regression slope and intercept from log-log fit to
/// power law of practice parameters.
///
/// If the regression was `ln(RT) = intercept + slope × ln(trial)`,
/// then `a = exp(intercept)` and `b = -slope`.
///
/// Returns `(initial_time, learning_rate)` for `T = a × N^(-b)`.
#[must_use]
#[inline]
pub fn loglog_regression_to_learning_rate(intercept: f64, slope: f64) -> (f64, f64) {
    (intercept.exp(), -slope)
}

/// Convert exponential smoothing parameter (alpha) to an approximate
/// memory stability parameter for the Ebbinghaus forgetting curve.
///
/// Higher smoothing alpha means faster adaptation (shorter memory).
/// `stability ≈ 1 / alpha` (in arbitrary time units).
#[must_use]
#[inline]
pub fn smoothing_alpha_to_stability(alpha: f64) -> f64 {
    if alpha <= 0.0 || !alpha.is_finite() {
        return f64::MAX;
    }
    1.0 / alpha
}

// ── Statistics → Decision Making ──────────────────────────────────────────

/// Convert a Bayesian posterior probability to odds ratio.
///
/// `odds = p / (1 - p)`
///
/// Useful for converting pramana's `bayes_theorem()` output to
/// bodh's odds-based decision models.
#[must_use]
#[inline]
pub fn posterior_to_odds(posterior: f64) -> f64 {
    if posterior >= 1.0 {
        return f64::MAX;
    }
    if posterior <= 0.0 {
        return 0.0;
    }
    posterior / (1.0 - posterior)
}

/// Convert logistic regression coefficient to probability change
/// per unit increase in the predictor.
///
/// For a logistic model with coefficient β, the odds ratio is `e^β`.
/// This returns the odds ratio, which can feed into prospect theory
/// or decision weighting models.
#[must_use]
#[inline]
pub fn logistic_coeff_to_odds_ratio(beta: f64) -> f64 {
    beta.exp()
}

// ── Statistics → Emotion/Arousal ──────────────────────────────────────────

/// Convert distribution variance to arousal uncertainty.
///
/// Higher variance in behavioral measures maps to higher uncertainty,
/// which can drive arousal in the Yerkes-Dodson model. Normalises
/// by a reference variance to produce a 0–1 arousal estimate.
#[must_use]
#[inline]
pub fn variance_to_arousal(variance: f64, reference_variance: f64) -> f64 {
    if reference_variance <= 0.0 || !reference_variance.is_finite() {
        return 0.0;
    }
    (variance / reference_variance).clamp(0.0, 1.0)
}

/// Convert z-score to a valence estimate.
///
/// Positive z → positive valence, negative z → negative valence.
/// Maps via tanh to keep output in \[-1, 1\].
#[must_use]
#[inline]
pub fn z_score_to_valence(z: f64) -> f64 {
    z.tanh()
}

// ── Statistics → Social ───────────────────────────────────────────────────

/// Convert ANOVA F-statistic to an effect size (eta-squared).
///
/// `η² = (F × df_between) / (F × df_between + df_within)`
///
/// Useful for quantifying the magnitude of between-group differences
/// in social conformity or group comparison studies.
#[must_use]
#[inline]
pub fn f_statistic_to_eta_squared(f: f64, df_between: usize, df_within: usize) -> f64 {
    if df_within == 0 || f <= 0.0 {
        return 0.0;
    }
    let num = f * df_between as f64;
    let denom = num + df_within as f64;
    if denom < 1e-15 {
        return 0.0;
    }
    (num / denom).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Signal detection ──

    #[test]
    fn test_t_to_d_prime() {
        // t=2.0, n=100: d' = 2/10 = 0.2
        let d = t_statistic_to_d_prime(2.0, 100);
        assert!((d - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_t_to_d_prime_zero_n() {
        assert!(t_statistic_to_d_prime(2.0, 0).abs() < 1e-10);
    }

    #[test]
    fn test_t_to_cohens_d() {
        // t=3.0, n1=30, n2=30, df=58: d = 6/√58 ≈ 0.788
        let d = t_statistic_to_cohens_d(3.0, 30, 30);
        assert!((d - 6.0 / 58.0_f64.sqrt()).abs() < 1e-10);
    }

    // ── Psychometrics ──

    #[test]
    fn test_variance_explained_clamps() {
        assert!((variance_explained_to_reliability(0.85) - 0.85).abs() < 1e-10);
        assert!((variance_explained_to_reliability(1.5) - 1.0).abs() < 1e-10);
        assert!((variance_explained_to_reliability(-0.1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_to_determination() {
        let (r_sq, _) = correlation_to_determination(0.7);
        assert!((r_sq - 0.49).abs() < 1e-10);
    }

    // ── Learning ──

    #[test]
    fn test_loglog_to_learning_rate() {
        // intercept = ln(10) ≈ 2.3026, slope = -0.3
        // → a = 10, b = 0.3
        let (a, b) = loglog_regression_to_learning_rate(10.0_f64.ln(), -0.3);
        assert!((a - 10.0).abs() < 1e-6);
        assert!((b - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_smoothing_to_stability() {
        assert!((smoothing_alpha_to_stability(0.1) - 10.0).abs() < 1e-10);
        assert!((smoothing_alpha_to_stability(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothing_to_stability_zero() {
        assert_eq!(smoothing_alpha_to_stability(0.0), f64::MAX);
    }

    // ── Decision making ──

    #[test]
    fn test_posterior_to_odds() {
        // p=0.75 → odds = 3.0
        assert!((posterior_to_odds(0.75) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_posterior_to_odds_extremes() {
        assert!(posterior_to_odds(0.0).abs() < 1e-10);
        assert_eq!(posterior_to_odds(1.0), f64::MAX);
    }

    #[test]
    fn test_logistic_coeff_to_odds_ratio() {
        // β=0 → OR=1.0 (no effect)
        assert!((logistic_coeff_to_odds_ratio(0.0) - 1.0).abs() < 1e-10);
        // β=ln(2) → OR=2.0
        assert!((logistic_coeff_to_odds_ratio(2.0_f64.ln()) - 2.0).abs() < 1e-10);
    }

    // ── Emotion/Arousal ──

    #[test]
    fn test_variance_to_arousal() {
        assert!((variance_to_arousal(5.0, 10.0) - 0.5).abs() < 1e-10);
        assert!((variance_to_arousal(20.0, 10.0) - 1.0).abs() < 1e-10); // clamped
    }

    #[test]
    fn test_z_score_to_valence() {
        assert!(z_score_to_valence(0.0).abs() < 1e-10);
        assert!(z_score_to_valence(3.0) > 0.9); // strongly positive
        assert!(z_score_to_valence(-3.0) < -0.9); // strongly negative
    }

    // ── Social ──

    #[test]
    fn test_f_to_eta_squared() {
        // F=5, df_b=2, df_w=27: η² = 10/(10+27) ≈ 0.270
        let eta = f_statistic_to_eta_squared(5.0, 2, 27);
        assert!((eta - 10.0 / 37.0).abs() < 1e-10);
    }

    #[test]
    fn test_f_to_eta_squared_zero() {
        assert!(f_statistic_to_eta_squared(0.0, 2, 27).abs() < 1e-10);
    }
}
