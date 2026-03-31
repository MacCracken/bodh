//! Stress — transactional stress model, coping, stress-performance.
//!
//! Lazarus & Folkman's transactional model of stress (1984),
//! coping resource depletion, and stress-performance relationships.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite, validate_positive};

// ---------------------------------------------------------------------------
// Lazarus Transactional Model (1984)
// ---------------------------------------------------------------------------

/// Primary appraisal: evaluation of a stimulus's personal significance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PrimaryAppraisal {
    /// Irrelevant — no personal significance.
    Irrelevant,
    /// Benign-positive — preserves or enhances well-being.
    BenignPositive,
    /// Stressful: harm/loss — damage already done.
    HarmLoss,
    /// Stressful: threat — anticipated future harm.
    Threat,
    /// Stressful: challenge — potential for growth.
    Challenge,
}

/// Secondary appraisal: evaluation of coping resources and options.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SecondaryAppraisal {
    /// Perceived control over the situation (0 = none, 1 = full).
    pub perceived_control: f64,
    /// Available coping resources (0 = depleted, 1 = abundant).
    pub coping_resources: f64,
    /// Confidence in coping ability (0 = none, 1 = total).
    pub self_efficacy: f64,
}

/// Compute stress intensity from primary and secondary appraisal.
///
/// `stress = threat_level × (1 − coping_capacity)`
///
/// where `threat_level` comes from the primary appraisal (0–1) and
/// `coping_capacity` is the mean of secondary appraisal components.
///
/// Stress is high when threat is high AND coping resources are low
/// (the core transactional principle).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[must_use = "returns the stress intensity without side effects"]
pub fn stress_intensity(primary: PrimaryAppraisal, secondary: &SecondaryAppraisal) -> Result<f64> {
    validate_finite(secondary.perceived_control, "perceived_control")?;
    validate_finite(secondary.coping_resources, "coping_resources")?;
    validate_finite(secondary.self_efficacy, "self_efficacy")?;

    let threat_level = match primary {
        PrimaryAppraisal::Irrelevant => 0.0,
        PrimaryAppraisal::BenignPositive => 0.0,
        PrimaryAppraisal::Challenge => 0.3,
        PrimaryAppraisal::Threat => 0.7,
        PrimaryAppraisal::HarmLoss => 1.0,
    };

    let coping =
        ((secondary.perceived_control + secondary.coping_resources + secondary.self_efficacy)
            / 3.0)
            .clamp(0.0, 1.0);

    Ok(threat_level * (1.0 - coping))
}

// ---------------------------------------------------------------------------
// Coping
// ---------------------------------------------------------------------------

/// Coping strategy type (Folkman & Lazarus, 1988).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum CopingStrategy {
    /// Problem-focused: address the source of stress directly.
    ProblemFocused,
    /// Emotion-focused: regulate emotional response to stress.
    EmotionFocused,
    /// Avoidance: disengage from the stressor.
    Avoidance,
    /// Social support: seek help from others.
    SocialSupport,
}

impl CopingStrategy {
    /// Typical effectiveness at reducing stress (0–1).
    ///
    /// Problem-focused is most effective when the situation is
    /// controllable; emotion-focused when it is not. These values
    /// represent the average case.
    #[inline]
    #[must_use]
    pub fn average_effectiveness(self) -> f64 {
        match self {
            Self::ProblemFocused => 0.8,
            Self::EmotionFocused => 0.6,
            Self::SocialSupport => 0.65,
            Self::Avoidance => 0.3,
        }
    }
}

/// Coping effectiveness adjusted for controllability.
///
/// Problem-focused coping works best for controllable stressors;
/// emotion-focused works best for uncontrollable ones.
///
/// `effectiveness = base × match_bonus`
///
/// where `match_bonus` rewards strategy–controllability fit.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if controllability is
/// outside \[0, 1\] or non-finite.
#[must_use = "returns the adjusted effectiveness without side effects"]
pub fn coping_effectiveness(strategy: CopingStrategy, controllability: f64) -> Result<f64> {
    validate_finite(controllability, "controllability")?;
    if !(0.0..=1.0).contains(&controllability) {
        return Err(BodhError::InvalidParameter(
            "controllability must be in [0, 1]".into(),
        ));
    }

    let base = strategy.average_effectiveness();
    let match_bonus = match strategy {
        CopingStrategy::ProblemFocused => 0.5 + 0.5 * controllability,
        CopingStrategy::EmotionFocused => 0.5 + 0.5 * (1.0 - controllability),
        CopingStrategy::SocialSupport => 0.8, // effective across contexts
        CopingStrategy::Avoidance => 0.3 + 0.4 * (1.0 - controllability),
    };
    Ok((base * match_bonus).clamp(0.0, 1.0))
}

// ---------------------------------------------------------------------------
// Resource Depletion (Conservation of Resources, Hobfoll, 1989)
// ---------------------------------------------------------------------------

/// Resource depletion under sustained stress.
///
/// `remaining = initial × e^(-depletion_rate × duration)`
///
/// Resources deplete exponentially under stress. Higher stress
/// intensity increases the depletion rate.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[inline]
#[must_use = "returns the remaining resources without side effects"]
pub fn resource_depletion(
    initial_resources: f64,
    stress_level: f64,
    duration: f64,
    depletion_rate: f64,
) -> Result<f64> {
    validate_positive(initial_resources, "initial_resources")?;
    validate_finite(stress_level, "stress_level")?;
    crate::error::validate_non_negative(duration, "duration")?;
    validate_positive(depletion_rate, "depletion_rate")?;

    let effective_rate = depletion_rate * stress_level.clamp(0.0, 1.0);
    Ok(initial_resources * (-effective_rate * duration).exp())
}

// ---------------------------------------------------------------------------
// Stress-Performance Relationship
// ---------------------------------------------------------------------------

/// Stress-performance curve (inverted-U / Yerkes-Dodson for stress).
///
/// `performance = peak × (1 − ((stress − optimal) / spread)²)`
///
/// Performance peaks at moderate stress and drops at both low
/// (underarousal) and high (overwhelm) stress levels.
///
/// This is the stress analogue of the Yerkes-Dodson law.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if `spread` is non-positive
/// or inputs are non-finite.
#[inline]
#[must_use = "returns the performance level without side effects"]
pub fn stress_performance(
    stress: f64,
    optimal_stress: f64,
    spread: f64,
    peak_performance: f64,
) -> Result<f64> {
    validate_finite(stress, "stress")?;
    validate_finite(optimal_stress, "optimal_stress")?;
    validate_positive(spread, "spread")?;
    validate_finite(peak_performance, "peak_performance")?;

    let deviation = stress - optimal_stress;
    let perf = peak_performance * (1.0 - (deviation / spread).powi(2));
    Ok(perf.max(0.0))
}

/// Burnout risk: probability of burnout from chronic stress exposure.
///
/// `risk = 1 − e^(-intensity × duration / resilience)`
///
/// Increases with stress intensity and duration, decreases with
/// resilience (individual protective factors).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[inline]
#[must_use = "returns the burnout risk without side effects"]
pub fn burnout_risk(stress_intensity: f64, duration: f64, resilience: f64) -> Result<f64> {
    validate_finite(stress_intensity, "stress_intensity")?;
    crate::error::validate_non_negative(duration, "duration")?;
    validate_positive(resilience, "resilience")?;

    let exposure = stress_intensity.max(0.0) * duration / resilience;
    Ok((1.0 - (-exposure).exp()).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Stress intensity --

    #[test]
    fn test_stress_irrelevant() {
        let secondary = SecondaryAppraisal {
            perceived_control: 0.5,
            coping_resources: 0.5,
            self_efficacy: 0.5,
        };
        let s = stress_intensity(PrimaryAppraisal::Irrelevant, &secondary).unwrap();
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn test_stress_threat_low_coping() {
        let secondary = SecondaryAppraisal {
            perceived_control: 0.1,
            coping_resources: 0.1,
            self_efficacy: 0.1,
        };
        let s = stress_intensity(PrimaryAppraisal::Threat, &secondary).unwrap();
        assert!(s > 0.5); // high stress
    }

    #[test]
    fn test_stress_threat_high_coping() {
        let secondary = SecondaryAppraisal {
            perceived_control: 0.9,
            coping_resources: 0.9,
            self_efficacy: 0.9,
        };
        let s = stress_intensity(PrimaryAppraisal::Threat, &secondary).unwrap();
        assert!(s < 0.1); // low stress despite threat
    }

    #[test]
    fn test_stress_challenge_moderate() {
        let secondary = SecondaryAppraisal {
            perceived_control: 0.5,
            coping_resources: 0.5,
            self_efficacy: 0.5,
        };
        let s = stress_intensity(PrimaryAppraisal::Challenge, &secondary).unwrap();
        assert!(s > 0.0 && s < 0.3); // challenge = mild stress
    }

    // -- Coping --

    #[test]
    fn test_problem_focused_controllable() {
        let eff = coping_effectiveness(CopingStrategy::ProblemFocused, 0.9).unwrap();
        assert!(eff > 0.7);
    }

    #[test]
    fn test_problem_focused_uncontrollable() {
        let eff = coping_effectiveness(CopingStrategy::ProblemFocused, 0.1).unwrap();
        let eff_high = coping_effectiveness(CopingStrategy::ProblemFocused, 0.9).unwrap();
        assert!(eff < eff_high); // less effective when uncontrollable
    }

    #[test]
    fn test_emotion_focused_uncontrollable() {
        let eff = coping_effectiveness(CopingStrategy::EmotionFocused, 0.1).unwrap();
        let eff_ctrl = coping_effectiveness(CopingStrategy::EmotionFocused, 0.9).unwrap();
        assert!(eff > eff_ctrl); // more effective when uncontrollable
    }

    #[test]
    fn test_avoidance_least_effective() {
        let avoid = coping_effectiveness(CopingStrategy::Avoidance, 0.5).unwrap();
        let problem = coping_effectiveness(CopingStrategy::ProblemFocused, 0.5).unwrap();
        assert!(problem > avoid);
    }

    // -- Resource depletion --

    #[test]
    fn test_resource_depletion_no_stress() {
        let r = resource_depletion(1.0, 0.0, 10.0, 0.5).unwrap();
        assert!((r - 1.0).abs() < 1e-10); // no depletion at zero stress
    }

    #[test]
    fn test_resource_depletion_over_time() {
        let r1 = resource_depletion(1.0, 0.8, 1.0, 0.5).unwrap();
        let r5 = resource_depletion(1.0, 0.8, 5.0, 0.5).unwrap();
        assert!(r1 > r5); // depletes over time
    }

    #[test]
    fn test_resource_depletion_known_value() {
        // stress=1, rate=1, t=1: remaining = e^(-1) ≈ 0.368
        let r = resource_depletion(1.0, 1.0, 1.0, 1.0).unwrap();
        assert!((r - (-1.0_f64).exp()).abs() < 1e-10);
    }

    // -- Stress-performance --

    #[test]
    fn test_stress_performance_optimal() {
        let p = stress_performance(0.5, 0.5, 0.5, 1.0).unwrap();
        assert!((p - 1.0).abs() < 1e-10); // peak at optimal
    }

    #[test]
    fn test_stress_performance_inverted_u() {
        let low = stress_performance(0.0, 0.5, 0.5, 1.0).unwrap();
        let opt = stress_performance(0.5, 0.5, 0.5, 1.0).unwrap();
        let high = stress_performance(1.0, 0.5, 0.5, 1.0).unwrap();
        assert!(opt > low);
        assert!(opt > high);
    }

    // -- Burnout --

    #[test]
    fn test_burnout_risk_zero_stress() {
        let r = burnout_risk(0.0, 10.0, 1.0).unwrap();
        assert!(r.abs() < 1e-10);
    }

    #[test]
    fn test_burnout_risk_increases() {
        let short = burnout_risk(0.8, 1.0, 1.0).unwrap();
        let long = burnout_risk(0.8, 10.0, 1.0).unwrap();
        assert!(long > short);
    }

    #[test]
    fn test_burnout_resilience_protects() {
        let fragile = burnout_risk(0.8, 5.0, 0.5).unwrap();
        let resilient = burnout_risk(0.8, 5.0, 5.0).unwrap();
        assert!(fragile > resilient);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_primary_appraisal_serde_roundtrip() {
        let a = PrimaryAppraisal::Threat;
        let json = serde_json::to_string(&a).unwrap();
        let back: PrimaryAppraisal = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }

    #[test]
    fn test_secondary_appraisal_serde_roundtrip() {
        let s = SecondaryAppraisal {
            perceived_control: 0.7,
            coping_resources: 0.5,
            self_efficacy: 0.8,
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: SecondaryAppraisal = serde_json::from_str(&json).unwrap();
        assert!((s.perceived_control - back.perceived_control).abs() < 1e-10);
    }

    #[test]
    fn test_coping_strategy_serde_roundtrip() {
        let c = CopingStrategy::SocialSupport;
        let json = serde_json::to_string(&c).unwrap();
        let back: CopingStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(c, back);
    }
}
