//! Motivation — self-determination theory, expectancy-value, flow state.
//!
//! Models for intrinsic/extrinsic motivation, basic psychological needs,
//! Eccles expectancy-value theory, and Csikszentmihalyi's flow model.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite};

// ---------------------------------------------------------------------------
// Self-Determination Theory (Deci & Ryan, 2000)
// ---------------------------------------------------------------------------

/// Basic psychological needs from Self-Determination Theory.
///
/// Each need is rated on a \[0, 1\] scale where 1 = fully satisfied.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BasicNeeds {
    /// Sense of volition and choice (0 = controlled, 1 = autonomous).
    pub autonomy: f64,
    /// Sense of effectiveness and mastery (0 = helpless, 1 = competent).
    pub competence: f64,
    /// Sense of connection and belonging (0 = isolated, 1 = connected).
    pub relatedness: f64,
}

impl BasicNeeds {
    /// Overall need satisfaction: mean of the three needs.
    #[inline]
    #[must_use]
    pub fn satisfaction(&self) -> f64 {
        (self.autonomy + self.competence + self.relatedness) / 3.0
    }

    /// The most deprived need (lowest value).
    #[must_use]
    pub fn most_deprived(&self) -> NeedType {
        if self.autonomy <= self.competence && self.autonomy <= self.relatedness {
            NeedType::Autonomy
        } else if self.competence <= self.relatedness {
            NeedType::Competence
        } else {
            NeedType::Relatedness
        }
    }
}

/// Which basic psychological need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum NeedType {
    /// Autonomy — volition and choice.
    Autonomy,
    /// Competence — effectiveness and mastery.
    Competence,
    /// Relatedness — connection and belonging.
    Relatedness,
}

/// Motivation type along the SDT continuum.
///
/// Ordered from most autonomous to most controlled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum MotivationType {
    /// No motivation or intention to act.
    Amotivation,
    /// Acting to obtain rewards or avoid punishment.
    ExternalRegulation,
    /// Acting to avoid guilt or gain ego-enhancement.
    IntrojectedRegulation,
    /// Personally valuing the activity.
    IdentifiedRegulation,
    /// Activity is congruent with self.
    IntegratedRegulation,
    /// Acting for inherent enjoyment.
    IntrinsicMotivation,
}

/// Predict motivation type from need satisfaction.
///
/// Higher need satisfaction → more autonomous motivation.
/// Thresholds based on SDT empirical findings.
#[must_use]
pub fn predict_motivation(needs: &BasicNeeds) -> MotivationType {
    let sat = needs.satisfaction();
    if sat < 0.15 {
        MotivationType::Amotivation
    } else if sat < 0.3 {
        MotivationType::ExternalRegulation
    } else if sat < 0.45 {
        MotivationType::IntrojectedRegulation
    } else if sat < 0.6 {
        MotivationType::IdentifiedRegulation
    } else if sat < 0.8 {
        MotivationType::IntegratedRegulation
    } else {
        MotivationType::IntrinsicMotivation
    }
}

/// Relative autonomy index (RAI).
///
/// Weighted sum of motivation subtypes where autonomous forms are
/// positive and controlled forms are negative.
///
/// `RAI = 2×intrinsic + identified − introjected − 2×external`
///
/// Higher RAI = more self-determined motivation.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any score is non-finite.
#[must_use = "returns the autonomy index without side effects"]
pub fn relative_autonomy_index(
    intrinsic: f64,
    identified: f64,
    introjected: f64,
    external: f64,
) -> Result<f64> {
    validate_finite(intrinsic, "intrinsic")?;
    validate_finite(identified, "identified")?;
    validate_finite(introjected, "introjected")?;
    validate_finite(external, "external")?;
    Ok(2.0 * intrinsic + identified - introjected - 2.0 * external)
}

// ---------------------------------------------------------------------------
// Expectancy-Value Theory (Eccles & Wigfield, 2002)
// ---------------------------------------------------------------------------

/// Expectancy-value model: motivation as the product of expectancy
/// and subjective task value.
///
/// `motivation = expectancy × value`
///
/// where `expectancy` is the belief about future success (0–1) and
/// `value` is the subjective importance of the task.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[inline]
#[must_use = "returns the motivation strength without side effects"]
pub fn expectancy_value(expectancy: f64, value: f64) -> Result<f64> {
    validate_finite(expectancy, "expectancy")?;
    validate_finite(value, "value")?;
    Ok(expectancy * value)
}

/// Task value components from Eccles' expectancy-value theory.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TaskValue {
    /// How interesting/enjoyable (intrinsic value), 0–1.
    pub intrinsic_value: f64,
    /// How important for identity/goals (attainment value), 0–1.
    pub attainment_value: f64,
    /// How useful for future plans (utility value), 0–1.
    pub utility_value: f64,
    /// Perceived cost of engaging (effort, opportunity cost), 0–1.
    pub cost: f64,
}

impl TaskValue {
    /// Net subjective value: weighted benefits minus cost.
    ///
    /// `net = (intrinsic + attainment + utility) / 3 − cost`
    #[inline]
    #[must_use]
    pub fn net_value(&self) -> f64 {
        (self.intrinsic_value + self.attainment_value + self.utility_value) / 3.0 - self.cost
    }
}

// ---------------------------------------------------------------------------
// Flow State (Csikszentmihalyi, 1990)
// ---------------------------------------------------------------------------

/// Flow state model: optimal experience when skill matches challenge.
///
/// Returns the flow intensity (0–1) based on the balance between
/// perceived challenge and perceived skill. Peak flow occurs when
/// both are high and well-matched.
///
/// `flow = match_factor × intensity`
///
/// where `match_factor = 1 − |challenge − skill|` and
/// `intensity = (challenge + skill) / 2`.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if challenge or skill is
/// outside \[0, 1\] or non-finite.
#[must_use = "returns the flow intensity without side effects"]
pub fn flow_state(challenge: f64, skill: f64) -> Result<f64> {
    validate_finite(challenge, "challenge")?;
    validate_finite(skill, "skill")?;
    if !(0.0..=1.0).contains(&challenge) {
        return Err(BodhError::InvalidParameter(
            "challenge must be in [0, 1]".into(),
        ));
    }
    if !(0.0..=1.0).contains(&skill) {
        return Err(BodhError::InvalidParameter(
            "skill must be in [0, 1]".into(),
        ));
    }

    let match_factor = 1.0 - (challenge - skill).abs();
    let intensity = (challenge + skill) / 2.0;
    Ok(match_factor * intensity)
}

/// Psychological state from challenge–skill balance.
///
/// Based on Csikszentmihalyi's experience fluctuation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum FlowChannel {
    /// Low challenge, low skill → apathy.
    Apathy,
    /// Low challenge, high skill → boredom/relaxation.
    Boredom,
    /// High challenge, low skill → anxiety.
    Anxiety,
    /// Matched challenge and skill → flow.
    Flow,
}

/// Classify the psychological state from challenge and skill levels.
///
/// Uses a midpoint threshold of 0.5 for high/low classification.
#[must_use]
pub fn classify_flow_channel(challenge: f64, skill: f64) -> FlowChannel {
    let high_challenge = challenge >= 0.5;
    let high_skill = skill >= 0.5;
    match (high_challenge, high_skill) {
        (true, true) => FlowChannel::Flow,
        (true, false) => FlowChannel::Anxiety,
        (false, true) => FlowChannel::Boredom,
        (false, false) => FlowChannel::Apathy,
    }
}

// ---------------------------------------------------------------------------
// Goal Gradient Effect (Hull, 1932)
// ---------------------------------------------------------------------------

/// Goal gradient effect: motivation increases as one approaches a goal.
///
/// `motivation = base × (1 + gradient × progress / (1 − progress + ε))`
///
/// where `progress` is in \[0, 1) and `gradient` controls the
/// acceleration (typically 0.5–2.0). As progress → 1, motivation
/// accelerates sharply (the "last mile" effect).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[inline]
#[must_use = "returns the motivation level without side effects"]
pub fn goal_gradient(progress: f64, base_motivation: f64, gradient: f64) -> Result<f64> {
    validate_finite(progress, "progress")?;
    validate_finite(base_motivation, "base_motivation")?;
    validate_finite(gradient, "gradient")?;
    if !(0.0..1.0).contains(&progress) {
        return Err(BodhError::InvalidParameter(
            "progress must be in [0, 1)".into(),
        ));
    }
    let boost = gradient * progress / (1.0 - progress + 1e-6);
    Ok(base_motivation * (1.0 + boost))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Basic Needs / SDT --

    #[test]
    fn test_needs_satisfaction() {
        let needs = BasicNeeds {
            autonomy: 0.8,
            competence: 0.6,
            relatedness: 0.7,
        };
        assert!((needs.satisfaction() - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_most_deprived() {
        let needs = BasicNeeds {
            autonomy: 0.9,
            competence: 0.2,
            relatedness: 0.5,
        };
        assert_eq!(needs.most_deprived(), NeedType::Competence);
    }

    #[test]
    fn test_predict_motivation_intrinsic() {
        let needs = BasicNeeds {
            autonomy: 0.9,
            competence: 0.9,
            relatedness: 0.9,
        };
        assert_eq!(
            predict_motivation(&needs),
            MotivationType::IntrinsicMotivation
        );
    }

    #[test]
    fn test_predict_motivation_amotivation() {
        let needs = BasicNeeds {
            autonomy: 0.05,
            competence: 0.05,
            relatedness: 0.05,
        };
        assert_eq!(predict_motivation(&needs), MotivationType::Amotivation);
    }

    #[test]
    fn test_predict_motivation_ordering() {
        // More autonomous motivation > more controlled
        assert!(MotivationType::IntrinsicMotivation > MotivationType::ExternalRegulation);
        assert!(MotivationType::IdentifiedRegulation > MotivationType::IntrojectedRegulation);
    }

    #[test]
    fn test_relative_autonomy_index() {
        // High intrinsic, low external → positive RAI.
        let rai = relative_autonomy_index(5.0, 4.0, 2.0, 1.0).unwrap();
        assert!(rai > 0.0);
        // Low intrinsic, high external → negative RAI.
        let rai2 = relative_autonomy_index(1.0, 2.0, 4.0, 5.0).unwrap();
        assert!(rai2 < 0.0);
    }

    #[test]
    fn test_rai_known_value() {
        // RAI = 2×3 + 2 − 1 − 2×0 = 6 + 2 − 1 − 0 = 7
        let rai = relative_autonomy_index(3.0, 2.0, 1.0, 0.0).unwrap();
        assert!((rai - 7.0).abs() < 1e-10);
    }

    // -- Expectancy-Value --

    #[test]
    fn test_expectancy_value_basic() {
        let m = expectancy_value(0.8, 10.0).unwrap();
        assert!((m - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_expectancy_value_zero() {
        let m = expectancy_value(0.0, 10.0).unwrap();
        assert!(m.abs() < 1e-10);
    }

    #[test]
    fn test_task_value_net() {
        let tv = TaskValue {
            intrinsic_value: 0.8,
            attainment_value: 0.6,
            utility_value: 0.4,
            cost: 0.3,
        };
        // net = (0.8+0.6+0.4)/3 − 0.3 = 0.6 − 0.3 = 0.3
        assert!((tv.net_value() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_task_value_high_cost() {
        let tv = TaskValue {
            intrinsic_value: 0.3,
            attainment_value: 0.3,
            utility_value: 0.3,
            cost: 0.8,
        };
        assert!(tv.net_value() < 0.0); // cost outweighs value
    }

    // -- Flow State --

    #[test]
    fn test_flow_peak() {
        // Perfect match at high levels → peak flow.
        let f = flow_state(0.9, 0.9).unwrap();
        assert!(f > 0.8);
    }

    #[test]
    fn test_flow_mismatch_reduces() {
        let matched = flow_state(0.8, 0.8).unwrap();
        let mismatched = flow_state(0.8, 0.2).unwrap();
        assert!(matched > mismatched);
    }

    #[test]
    fn test_flow_low_both() {
        // Low challenge + low skill → low flow even if matched.
        let f = flow_state(0.1, 0.1).unwrap();
        assert!(f < 0.2);
    }

    #[test]
    fn test_flow_invalid() {
        assert!(flow_state(-0.1, 0.5).is_err());
        assert!(flow_state(0.5, 1.5).is_err());
    }

    #[test]
    fn test_classify_flow_channel() {
        assert_eq!(classify_flow_channel(0.8, 0.8), FlowChannel::Flow);
        assert_eq!(classify_flow_channel(0.8, 0.2), FlowChannel::Anxiety);
        assert_eq!(classify_flow_channel(0.2, 0.8), FlowChannel::Boredom);
        assert_eq!(classify_flow_channel(0.2, 0.2), FlowChannel::Apathy);
    }

    // -- Goal Gradient --

    #[test]
    fn test_goal_gradient_increases() {
        let early = goal_gradient(0.2, 1.0, 1.0).unwrap();
        let late = goal_gradient(0.8, 1.0, 1.0).unwrap();
        assert!(late > early);
    }

    #[test]
    fn test_goal_gradient_zero_progress() {
        let m = goal_gradient(0.0, 1.0, 1.0).unwrap();
        assert!((m - 1.0).abs() < 0.01); // ~base motivation at start
    }

    #[test]
    fn test_goal_gradient_invalid() {
        assert!(goal_gradient(1.0, 1.0, 1.0).is_err()); // progress = 1.0 excluded
        assert!(goal_gradient(-0.1, 1.0, 1.0).is_err());
    }

    // -- Known values --

    #[test]
    fn test_expectancy_value_known() {
        // 0.7 × 8.0 = 5.6
        let m = expectancy_value(0.7, 8.0).unwrap();
        assert!((m - 5.6).abs() < 1e-10);
    }

    #[test]
    fn test_flow_state_known_value() {
        // challenge=0.8, skill=0.6: match=1−0.2=0.8, intensity=0.7
        // flow = 0.8 × 0.7 = 0.56
        let f = flow_state(0.8, 0.6).unwrap();
        assert!((f - 0.56).abs() < 1e-10);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_basic_needs_serde_roundtrip() {
        let needs = BasicNeeds {
            autonomy: 0.7,
            competence: 0.5,
            relatedness: 0.8,
        };
        let json = serde_json::to_string(&needs).unwrap();
        let back: BasicNeeds = serde_json::from_str(&json).unwrap();
        assert!((needs.autonomy - back.autonomy).abs() < 1e-10);
    }

    #[test]
    fn test_need_type_serde_roundtrip() {
        let n = NeedType::Competence;
        let json = serde_json::to_string(&n).unwrap();
        let back: NeedType = serde_json::from_str(&json).unwrap();
        assert_eq!(n, back);
    }

    #[test]
    fn test_motivation_type_serde_roundtrip() {
        let m = MotivationType::IdentifiedRegulation;
        let json = serde_json::to_string(&m).unwrap();
        let back: MotivationType = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn test_task_value_serde_roundtrip() {
        let tv = TaskValue {
            intrinsic_value: 0.8,
            attainment_value: 0.6,
            utility_value: 0.4,
            cost: 0.2,
        };
        let json = serde_json::to_string(&tv).unwrap();
        let back: TaskValue = serde_json::from_str(&json).unwrap();
        assert!((tv.cost - back.cost).abs() < 1e-10);
    }

    #[test]
    fn test_flow_channel_serde_roundtrip() {
        let f = FlowChannel::Anxiety;
        let json = serde_json::to_string(&f).unwrap();
        let back: FlowChannel = serde_json::from_str(&json).unwrap();
        assert_eq!(f, back);
    }
}
