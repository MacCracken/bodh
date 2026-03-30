//! Perception — signal detection theory, Gestalt principles.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite};

/// Signal detection theory parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalDetection {
    /// Proportion of signal trials correctly identified (0-1).
    pub hit_rate: f64,
    /// Proportion of noise trials incorrectly identified as signal (0-1).
    pub false_alarm_rate: f64,
}

impl SignalDetection {
    /// Create a new signal detection measurement.
    ///
    /// # Errors
    ///
    /// Returns [`BodhError::InvalidParameter`] if rates are outside (0, 1).
    pub fn new(hit_rate: f64, false_alarm_rate: f64) -> Result<Self> {
        validate_finite(hit_rate, "hit_rate")?;
        validate_finite(false_alarm_rate, "false_alarm_rate")?;
        if hit_rate <= 0.0 || hit_rate >= 1.0 {
            return Err(BodhError::InvalidParameter(
                "hit_rate must be in (0, 1) exclusive for d' calculation".into(),
            ));
        }
        if false_alarm_rate <= 0.0 || false_alarm_rate >= 1.0 {
            return Err(BodhError::InvalidParameter(
                "false_alarm_rate must be in (0, 1) exclusive for d' calculation".into(),
            ));
        }
        Ok(Self {
            hit_rate,
            false_alarm_rate,
        })
    }
}

/// Compute d-prime (d'): sensitivity measure in signal detection theory.
///
/// `d' = z(hit_rate) - z(false_alarm_rate)`
///
/// where `z` is the inverse of the standard normal CDF (probit function).
///
/// d' = 0 means chance performance (cannot distinguish signal from noise).
/// d' > 0 means above-chance sensitivity.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if rates are outside (0, 1).
#[must_use = "returns d-prime without side effects"]
pub fn d_prime(hit_rate: f64, false_alarm_rate: f64) -> Result<f64> {
    let sd = SignalDetection::new(hit_rate, false_alarm_rate)?;
    let z_hit = probit(sd.hit_rate);
    let z_fa = probit(sd.false_alarm_rate);
    Ok(z_hit - z_fa)
}

/// Response bias (criterion c) in signal detection theory.
///
/// `c = -0.5 * (z(hit_rate) + z(false_alarm_rate))`
///
/// c = 0: no bias. c > 0: conservative (say "no" more). c < 0: liberal.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if rates are outside (0, 1).
#[must_use = "returns the criterion without side effects"]
pub fn criterion_c(hit_rate: f64, false_alarm_rate: f64) -> Result<f64> {
    let sd = SignalDetection::new(hit_rate, false_alarm_rate)?;
    let z_hit = probit(sd.hit_rate);
    let z_fa = probit(sd.false_alarm_rate);
    Ok(-0.5 * (z_hit + z_fa))
}

/// Approximate probit function (inverse standard normal CDF).
///
/// Uses the rational approximation from Abramowitz & Stegun (26.2.23)
/// with maximum error < 4.5e-4.
#[inline]
fn probit(p: f64) -> f64 {
    // Ensure p is in (0, 1)
    let p = p.clamp(1e-10, 1.0 - 1e-10);

    if p < 0.5 {
        -rational_approx((-2.0 * p.ln()).sqrt())
    } else {
        rational_approx((-2.0 * (1.0 - p).ln()).sqrt())
    }
}

/// Rational approximation for the probit function.
#[inline]
fn rational_approx(t: f64) -> f64 {
    // Coefficients from Abramowitz & Stegun
    const C0: f64 = 2.515517;
    const C1: f64 = 0.802853;
    const C2: f64 = 0.010328;
    const D1: f64 = 1.432788;
    const D2: f64 = 0.189269;
    const D3: f64 = 0.001308;

    t - (C0 + C1 * t + C2 * t * t) / (1.0 + D1 * t + D2 * t * t + D3 * t * t * t)
}

/// Gestalt grouping principles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum GestaltPrinciple {
    /// Elements near each other are grouped together.
    Proximity,
    /// Similar elements are grouped together.
    Similarity,
    /// Incomplete figures are perceived as complete.
    Closure,
    /// Elements forming smooth lines are grouped.
    Continuity,
    /// Elements moving together are grouped.
    CommonFate,
    /// Elements in an enclosed region are grouped.
    CommonRegion,
    /// Connected elements are grouped.
    Connectedness,
}

impl GestaltPrinciple {
    /// Relative strength of the grouping principle (empirical ordering).
    #[inline]
    #[must_use]
    pub fn relative_strength(self) -> f64 {
        match self {
            Self::Connectedness => 1.0,
            Self::CommonRegion => 0.9,
            Self::CommonFate => 0.8,
            Self::Proximity => 0.7,
            Self::Similarity => 0.6,
            Self::Continuity => 0.5,
            Self::Closure => 0.4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_prime_chance() {
        // When hit rate equals false alarm rate, d' should be ~0.
        let d = d_prime(0.5, 0.5).unwrap();
        assert!(d.abs() < 0.01);
    }

    #[test]
    fn test_d_prime_good_performance() {
        // High hit rate + low false alarm rate = positive d'.
        let d = d_prime(0.9, 0.1).unwrap();
        assert!(d > 2.0);
    }

    #[test]
    fn test_d_prime_known_reference() {
        // d'(0.75, 0.25) ≈ 1.349 (from standard normal z-tables:
        // z(0.75) ≈ 0.6745, z(0.25) ≈ -0.6745, d' ≈ 1.349).
        let d = d_prime(0.75, 0.25).unwrap();
        assert!((d - 1.349).abs() < 0.01);
    }

    #[test]
    fn test_criterion_no_bias() {
        // Symmetric performance should have ~0 bias.
        let c = criterion_c(0.8, 0.2).unwrap();
        assert!(c.abs() < 0.1);
    }

    #[test]
    fn test_criterion_conservative() {
        // Low hit rate + low false alarm = conservative (c > 0).
        let c = criterion_c(0.3, 0.05).unwrap();
        assert!(c > 0.0);
    }

    #[test]
    fn test_d_prime_invalid_rates() {
        assert!(d_prime(0.0, 0.5).is_err());
        assert!(d_prime(1.0, 0.5).is_err());
        assert!(d_prime(0.5, 0.0).is_err());
        assert!(d_prime(0.5, 1.0).is_err());
    }

    #[test]
    fn test_gestalt_principle_strength() {
        let conn = GestaltPrinciple::Connectedness;
        let closure = GestaltPrinciple::Closure;
        assert!(conn.relative_strength() > closure.relative_strength());
    }

    #[test]
    fn test_gestalt_serde_roundtrip() {
        let g = GestaltPrinciple::CommonFate;
        let json = serde_json::to_string(&g).unwrap();
        let back: GestaltPrinciple = serde_json::from_str(&json).unwrap();
        assert_eq!(g, back);
    }

    #[test]
    fn test_probit_accuracy() {
        // Verify probit via d_prime: z(0.975) ≈ 1.96, z(0.5) = 0.
        // d'(0.975, 0.5) = z(0.975) - z(0.5) ≈ 1.96 - 0 = 1.96.
        let d = d_prime(0.975, 0.5).unwrap();
        assert!((d - 1.96).abs() < 0.001); // A&S 26.2.23 error < 4.5e-4
    }

    #[test]
    fn test_d_prime_reference_90_10() {
        // d'(0.9, 0.1): z(0.9) ≈ 1.2816, z(0.1) ≈ -1.2816, d' ≈ 2.563.
        let d = d_prime(0.9, 0.1).unwrap();
        assert!((d - 2.563).abs() < 0.01);
    }

    #[test]
    fn test_signal_detection_serde_roundtrip() {
        let sd = SignalDetection::new(0.8, 0.2).unwrap();
        let json = serde_json::to_string(&sd).unwrap();
        let back: SignalDetection = serde_json::from_str(&json).unwrap();
        assert!((sd.hit_rate - back.hit_rate).abs() < 1e-10);
    }
}
