//! Learning — memory, forgetting curves, spaced repetition, conditioning.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite, validate_non_negative, validate_positive};

/// Ebbinghaus forgetting curve: retention as a function of time.
///
/// `R = e^(-t / S)`
///
/// where `t` is time elapsed and `S` is memory stability (higher = slower
/// forgetting). At `t = 0`, `R = 1.0` (perfect recall).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if time is negative or stability is non-positive.
#[inline]
#[must_use = "returns the retention value without side effects"]
pub fn ebbinghaus_forgetting(time: f64, stability: f64) -> Result<f64> {
    validate_non_negative(time, "time")?;
    validate_positive(stability, "stability")?;
    Ok((-time / stability).exp())
}

/// Ebbinghaus forgetting curve with initial strength.
///
/// `R = strength * e^(-t / S)`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if parameters are invalid.
#[inline]
#[must_use = "returns the retention value without side effects"]
pub fn ebbinghaus_forgetting_full(strength: f64, time: f64, stability: f64) -> Result<f64> {
    validate_finite(strength, "strength")?;
    let retention = ebbinghaus_forgetting(time, stability)?;
    Ok(strength * retention)
}

/// Geometric spaced repetition interval.
///
/// `interval = base_interval * ease^(repetition - 1)`
///
/// where `repetition` is the review number (1-indexed) and `ease` is
/// the ease factor (typically 2.5 for a well-learned item).
///
/// This is a geometric growth model commonly used in modern spaced
/// repetition systems. It is *not* the literal SM-2 algorithm (which
/// uses hardcoded first intervals of 1 and 6 days), but captures the
/// same exponentially-growing interval principle.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if ease is non-positive.
#[inline]
#[must_use = "returns the interval without side effects"]
pub fn spaced_repetition_interval(repetition: u32, ease: f64) -> Result<f64> {
    validate_positive(ease, "ease")?;
    if repetition == 0 {
        return Err(BodhError::InvalidParameter(
            "repetition must be at least 1".into(),
        ));
    }
    let base = 1.0; // 1 day base interval
    Ok(base * ease.powi(repetition as i32 - 1))
}

/// Learning curve: performance as a function of practice trials.
///
/// Power law of practice: `T = a * N^(-b)`
///
/// where `N` is trial number, `a` is initial performance time, and `b` is
/// the learning rate (typically 0.2-0.5).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if trial is 0 or parameters are invalid.
#[inline]
#[must_use = "returns the performance time without side effects"]
pub fn power_law_of_practice(trial: u32, initial_time: f64, learning_rate: f64) -> Result<f64> {
    if trial == 0 {
        return Err(BodhError::InvalidParameter(
            "trial must be at least 1".into(),
        ));
    }
    validate_positive(initial_time, "initial_time")?;
    validate_positive(learning_rate, "learning_rate")?;
    Ok(initial_time * (trial as f64).powf(-learning_rate))
}

/// Reinforcement schedule types from operant conditioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum ReinforcementSchedule {
    /// Reinforcement after a fixed number of responses.
    FixedRatio,
    /// Reinforcement after a variable number of responses (average).
    VariableRatio,
    /// Reinforcement after a fixed time interval.
    FixedInterval,
    /// Reinforcement after a variable time interval (average).
    VariableInterval,
    /// Every response is reinforced.
    Continuous,
}

impl ReinforcementSchedule {
    /// Relative resistance to extinction (higher = more resistant).
    ///
    /// Variable schedules produce higher resistance to extinction than fixed.
    #[inline]
    #[must_use]
    pub fn extinction_resistance(self) -> f64 {
        match self {
            Self::Continuous => 0.2,
            Self::FixedRatio => 0.5,
            Self::FixedInterval => 0.4,
            Self::VariableRatio => 0.9,
            Self::VariableInterval => 0.8,
        }
    }
}

/// Classical conditioning strength (Rescorla-Wagner model).
///
/// `delta_V = alpha * beta * (lambda - V)`
///
/// where `alpha` is CS salience (0-1), `beta` is US intensity (0-1),
/// `lambda` is maximum conditioning (1.0), and `V` is current associative
/// strength.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if parameters are non-finite.
#[inline]
#[must_use = "returns the change in associative strength without side effects"]
pub fn rescorla_wagner(alpha: f64, beta: f64, lambda: f64, current_v: f64) -> Result<f64> {
    validate_finite(alpha, "alpha")?;
    validate_finite(beta, "beta")?;
    validate_finite(lambda, "lambda")?;
    validate_finite(current_v, "current_v")?;
    Ok(alpha * beta * (lambda - current_v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ebbinghaus_at_t0() {
        // At t=0, retention should be 1.0.
        let r = ebbinghaus_forgetting(0.0, 1.0).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ebbinghaus_decay() {
        // At t=stability, retention should be e^(-1) ≈ 0.368.
        let r = ebbinghaus_forgetting(1.0, 1.0).unwrap();
        assert!((r - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_ebbinghaus_full_with_strength() {
        let r = ebbinghaus_forgetting_full(0.8, 0.0, 1.0).unwrap();
        assert!((r - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_ebbinghaus_monotonic_decay() {
        let r1 = ebbinghaus_forgetting(1.0, 2.0).unwrap();
        let r2 = ebbinghaus_forgetting(2.0, 2.0).unwrap();
        let r3 = ebbinghaus_forgetting(5.0, 2.0).unwrap();
        assert!(r1 > r2);
        assert!(r2 > r3);
    }

    #[test]
    fn test_spaced_repetition_first() {
        let interval = spaced_repetition_interval(1, 2.5).unwrap();
        assert!((interval - 1.0).abs() < 1e-10); // base interval
    }

    #[test]
    fn test_spaced_repetition_grows() {
        let i1 = spaced_repetition_interval(1, 2.5).unwrap();
        let i2 = spaced_repetition_interval(2, 2.5).unwrap();
        let i3 = spaced_repetition_interval(3, 2.5).unwrap();
        assert!(i2 > i1);
        assert!(i3 > i2);
    }

    #[test]
    fn test_power_law_of_practice() {
        // First trial should be initial_time.
        let t = power_law_of_practice(1, 10.0, 0.3).unwrap();
        assert!((t - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_power_law_improvement() {
        let t1 = power_law_of_practice(1, 10.0, 0.3).unwrap();
        let t10 = power_law_of_practice(10, 10.0, 0.3).unwrap();
        assert!(t10 < t1); // faster with practice
    }

    #[test]
    fn test_reinforcement_schedule_extinction() {
        let vr = ReinforcementSchedule::VariableRatio;
        let cont = ReinforcementSchedule::Continuous;
        assert!(vr.extinction_resistance() > cont.extinction_resistance());
    }

    #[test]
    fn test_rescorla_wagner_acquisition() {
        // Starting from V=0, learning should be positive.
        let dv = rescorla_wagner(0.5, 0.5, 1.0, 0.0).unwrap();
        assert!((dv - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_rescorla_wagner_asymptote() {
        // At V=lambda, no further learning.
        let dv = rescorla_wagner(0.5, 0.5, 1.0, 1.0).unwrap();
        assert!((dv - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reinforcement_schedule_serde_roundtrip() {
        let sched = ReinforcementSchedule::VariableRatio;
        let json = serde_json::to_string(&sched).unwrap();
        let back: ReinforcementSchedule = serde_json::from_str(&json).unwrap();
        assert_eq!(sched, back);
    }
}
