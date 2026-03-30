//! Emotion — affect models, appraisal, emotion-cognition interaction.
//!
//! Russell's circumplex model (valence × arousal), Ekman's basic emotions,
//! Scherer's appraisal dimensions, and Gross's emotion regulation model.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite};

// ---------------------------------------------------------------------------
// Russell's Circumplex Model
// ---------------------------------------------------------------------------

/// Affective state in Russell's circumplex model (1980).
///
/// Two-dimensional representation: valence (pleasant–unpleasant) and
/// arousal (activated–deactivated), each in \[-1, 1\].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Affect {
    /// Pleasant (+1) to unpleasant (−1).
    pub valence: f64,
    /// High arousal (+1) to low arousal (−1).
    pub arousal: f64,
}

impl Affect {
    /// Create a new affect state, clamping to \[-1, 1\].
    ///
    /// # Errors
    ///
    /// Returns [`BodhError::InvalidParameter`] if either value is non-finite.
    pub fn new(valence: f64, arousal: f64) -> Result<Self> {
        validate_finite(valence, "valence")?;
        validate_finite(arousal, "arousal")?;
        Ok(Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
        })
    }

    /// Euclidean distance between two affect states in the circumplex.
    #[inline]
    #[must_use]
    pub fn distance(self, other: Self) -> f64 {
        let dv = self.valence - other.valence;
        let da = self.arousal - other.arousal;
        (dv * dv + da * da).sqrt()
    }

    /// Intensity of the affective state (distance from neutral origin).
    #[inline]
    #[must_use]
    pub fn intensity(self) -> f64 {
        (self.valence * self.valence + self.arousal * self.arousal).sqrt()
    }

    /// Angle in the circumplex (radians), measured counter-clockwise
    /// from the positive valence axis.
    #[inline]
    #[must_use]
    pub fn angle(self) -> f64 {
        self.arousal.atan2(self.valence)
    }

    /// Linear blend between two affect states.
    ///
    /// `t = 0.0` returns `self`, `t = 1.0` returns `other`.
    ///
    /// # Errors
    ///
    /// Returns [`BodhError::InvalidParameter`] if `t` is non-finite.
    #[must_use = "returns the blended affect without side effects"]
    pub fn blend(self, other: Self, t: f64) -> Result<Self> {
        validate_finite(t, "t")?;
        let t = t.clamp(0.0, 1.0);
        Ok(Self {
            valence: self.valence + t * (other.valence - self.valence),
            arousal: self.arousal + t * (other.arousal - self.arousal),
        })
    }
}

// ---------------------------------------------------------------------------
// Ekman's Basic Emotions
// ---------------------------------------------------------------------------

/// Ekman's six basic emotions (1992).
///
/// Each has a canonical position in the circumplex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BasicEmotion {
    /// Happiness — high valence, moderate arousal.
    Happiness,
    /// Sadness — low valence, low arousal.
    Sadness,
    /// Anger — low valence, high arousal.
    Anger,
    /// Fear — low valence, high arousal.
    Fear,
    /// Disgust — low valence, moderate arousal.
    Disgust,
    /// Surprise — neutral valence, high arousal.
    Surprise,
}

impl BasicEmotion {
    /// Canonical circumplex position for this emotion.
    ///
    /// Values based on empirical affect grid placements
    /// (Russell & Barrett, 1999).
    #[must_use]
    pub fn canonical_affect(self) -> Affect {
        match self {
            Self::Happiness => Affect {
                valence: 0.8,
                arousal: 0.3,
            },
            Self::Sadness => Affect {
                valence: -0.7,
                arousal: -0.5,
            },
            Self::Anger => Affect {
                valence: -0.6,
                arousal: 0.7,
            },
            Self::Fear => Affect {
                valence: -0.7,
                arousal: 0.8,
            },
            Self::Disgust => Affect {
                valence: -0.8,
                arousal: 0.2,
            },
            Self::Surprise => Affect {
                valence: 0.1,
                arousal: 0.9,
            },
        }
    }
}

/// Classify an affect state to the nearest basic emotion.
///
/// Uses Euclidean distance in the circumplex to find the closest
/// canonical emotion position.
#[must_use = "returns the classified emotion without side effects"]
pub fn classify_emotion(affect: Affect) -> BasicEmotion {
    use BasicEmotion::*;
    let emotions = [Happiness, Sadness, Anger, Fear, Disgust, Surprise];
    let mut best = Happiness;
    let mut best_dist = f64::MAX;
    for e in emotions {
        let d = affect.distance(e.canonical_affect());
        if d < best_dist {
            best_dist = d;
            best = e;
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Scherer's Appraisal Dimensions
// ---------------------------------------------------------------------------

/// Stimulus evaluation checks from Scherer's Component Process Model (2001).
///
/// Each dimension is evaluated on a \[-1, 1\] scale.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AppraisalDimensions {
    /// Novelty/unexpectedness (−1 expected, +1 novel).
    pub novelty: f64,
    /// Intrinsic pleasantness (−1 unpleasant, +1 pleasant).
    pub pleasantness: f64,
    /// Goal relevance/conduciveness (−1 obstructive, +1 conducive).
    pub goal_conduciveness: f64,
    /// Coping potential (−1 no control, +1 full control).
    pub coping_potential: f64,
    /// Norm compatibility (−1 violates norms, +1 norm-consistent).
    pub norm_compatibility: f64,
}

/// Map appraisal dimensions to an affect state.
///
/// Simplified mapping based on Scherer (2001):
/// - Valence ← weighted sum of pleasantness and goal conduciveness.
/// - Arousal ← weighted sum of novelty and inverse coping potential.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any dimension is non-finite.
#[must_use = "returns the appraised affect without side effects"]
pub fn appraise(dims: &AppraisalDimensions) -> Result<Affect> {
    validate_finite(dims.novelty, "novelty")?;
    validate_finite(dims.pleasantness, "pleasantness")?;
    validate_finite(dims.goal_conduciveness, "goal_conduciveness")?;
    validate_finite(dims.coping_potential, "coping_potential")?;
    validate_finite(dims.norm_compatibility, "norm_compatibility")?;

    let valence = (0.5 * dims.pleasantness + 0.5 * dims.goal_conduciveness).clamp(-1.0, 1.0);
    let arousal = (0.6 * dims.novelty + 0.4 * (1.0 - dims.coping_potential)).clamp(-1.0, 1.0);

    Ok(Affect { valence, arousal })
}

// ---------------------------------------------------------------------------
// Gross's Process Model of Emotion Regulation
// ---------------------------------------------------------------------------

/// Emotion regulation strategy (Gross, 1998).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum RegulationStrategy {
    /// Situation selection — avoid/approach emotion-eliciting situations.
    SituationSelection,
    /// Situation modification — change the situation.
    SituationModification,
    /// Attentional deployment — redirect attention (distraction, rumination).
    AttentionalDeployment,
    /// Cognitive change — reappraisal, reframing.
    CognitiveChange,
    /// Response modulation — suppress or amplify expression.
    ResponseModulation,
}

impl RegulationStrategy {
    /// Typical effectiveness of the strategy at reducing negative affect.
    ///
    /// Based on meta-analytic findings (Webb, Miles, & Sheeran, 2012).
    /// Higher = more effective. Cognitive change (reappraisal) is
    /// consistently the most effective antecedent-focused strategy.
    #[inline]
    #[must_use]
    pub fn effectiveness(self) -> f64 {
        match self {
            Self::SituationSelection => 0.6,
            Self::SituationModification => 0.5,
            Self::AttentionalDeployment => 0.45,
            Self::CognitiveChange => 0.85,
            Self::ResponseModulation => 0.3,
        }
    }
}

/// Apply an emotion regulation strategy to an affect state.
///
/// Moves the affect state toward neutral by a fraction proportional
/// to the strategy's effectiveness and the regulation `effort` (0–1).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if effort is non-finite.
#[must_use = "returns the regulated affect without side effects"]
pub fn regulate(affect: Affect, strategy: RegulationStrategy, effort: f64) -> Result<Affect> {
    validate_finite(effort, "effort")?;
    let effort = effort.clamp(0.0, 1.0);
    let reduction = strategy.effectiveness() * effort;

    Ok(Affect {
        valence: affect.valence * (1.0 - reduction),
        arousal: affect.arousal * (1.0 - reduction),
    })
}

// ---------------------------------------------------------------------------
// Emotion–Cognition Interaction
// ---------------------------------------------------------------------------

/// Yerkes–Dodson law: performance as a function of arousal.
///
/// `performance = 1 - (arousal - optimal)^2 / spread^2`
///
/// Inverted-U relationship: performance peaks at `optimal` arousal
/// and drops off at both low and high arousal. `spread` controls the
/// width of the performance curve (task difficulty).
///
/// Simple tasks have higher optimal arousal; complex tasks have lower.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if spread is non-positive or inputs non-finite.
#[inline]
#[must_use = "returns the performance level without side effects"]
pub fn yerkes_dodson(arousal: f64, optimal: f64, spread: f64) -> Result<f64> {
    validate_finite(arousal, "arousal")?;
    validate_finite(optimal, "optimal")?;
    crate::error::validate_positive(spread, "spread")?;
    let deviation = arousal - optimal;
    let perf = 1.0 - (deviation * deviation) / (spread * spread);
    Ok(perf.max(0.0))
}

/// Mood-congruent memory bias: probability boost for retrieving
/// memories that match the current affective state.
///
/// `bias = base_probability * (1 + congruence_weight * similarity)`
///
/// where `similarity` is 1.0 − distance between the current affect and
/// the memory's encoded affect, normalised to \[0, 1\].
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or weight is negative.
#[must_use = "returns the biased retrieval probability without side effects"]
pub fn mood_congruent_bias(
    base_probability: f64,
    current_affect: Affect,
    memory_affect: Affect,
    congruence_weight: f64,
) -> Result<f64> {
    validate_finite(base_probability, "base_probability")?;
    validate_finite(congruence_weight, "congruence_weight")?;
    if congruence_weight < 0.0 {
        return Err(BodhError::InvalidParameter(
            "congruence_weight must be non-negative".into(),
        ));
    }

    // Max possible distance in [-1,1]×[-1,1] is 2√2 ≈ 2.828.
    let max_dist = 2.0_f64.sqrt() * 2.0;
    let dist = current_affect.distance(memory_affect);
    let similarity = 1.0 - (dist / max_dist).min(1.0);
    let biased = base_probability * (1.0 + congruence_weight * similarity);
    Ok(biased.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Affect --

    #[test]
    fn test_affect_new_clamps() {
        let a = Affect::new(2.0, -3.0).unwrap();
        assert!((a.valence - 1.0).abs() < 1e-10);
        assert!((a.arousal - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_affect_invalid() {
        assert!(Affect::new(f64::NAN, 0.0).is_err());
    }

    #[test]
    fn test_affect_distance_same() {
        let a = Affect::new(0.5, 0.5).unwrap();
        assert!(a.distance(a) < 1e-10);
    }

    #[test]
    fn test_affect_distance_opposites() {
        let a = Affect::new(1.0, 1.0).unwrap();
        let b = Affect::new(-1.0, -1.0).unwrap();
        let d = a.distance(b);
        assert!((d - 2.0 * 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_affect_intensity() {
        let a = Affect::new(0.0, 0.0).unwrap();
        assert!(a.intensity() < 1e-10);
        let b = Affect::new(1.0, 0.0).unwrap();
        assert!((b.intensity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_affect_blend() {
        let a = Affect::new(0.0, 0.0).unwrap();
        let b = Affect::new(1.0, 1.0).unwrap();
        let mid = a.blend(b, 0.5).unwrap();
        assert!((mid.valence - 0.5).abs() < 1e-10);
        assert!((mid.arousal - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_affect_blend_endpoints() {
        let a = Affect::new(-0.5, 0.3).unwrap();
        let b = Affect::new(0.8, -0.2).unwrap();
        let at0 = a.blend(b, 0.0).unwrap();
        let at1 = a.blend(b, 1.0).unwrap();
        assert!((at0.valence - a.valence).abs() < 1e-10);
        assert!((at1.valence - b.valence).abs() < 1e-10);
    }

    #[test]
    fn test_affect_serde_roundtrip() {
        let a = Affect::new(0.5, -0.3).unwrap();
        let json = serde_json::to_string(&a).unwrap();
        let back: Affect = serde_json::from_str(&json).unwrap();
        assert!((a.valence - back.valence).abs() < 1e-10);
        assert!((a.arousal - back.arousal).abs() < 1e-10);
    }

    // -- Basic Emotions --

    #[test]
    fn test_classify_happiness() {
        let affect = Affect {
            valence: 0.9,
            arousal: 0.2,
        };
        assert_eq!(classify_emotion(affect), BasicEmotion::Happiness);
    }

    #[test]
    fn test_classify_sadness() {
        let affect = Affect {
            valence: -0.6,
            arousal: -0.6,
        };
        assert_eq!(classify_emotion(affect), BasicEmotion::Sadness);
    }

    #[test]
    fn test_classify_anger() {
        let affect = Affect {
            valence: -0.5,
            arousal: 0.6,
        };
        assert_eq!(classify_emotion(affect), BasicEmotion::Anger);
    }

    #[test]
    fn test_basic_emotion_serde_roundtrip() {
        let e = BasicEmotion::Fear;
        let json = serde_json::to_string(&e).unwrap();
        let back: BasicEmotion = serde_json::from_str(&json).unwrap();
        assert_eq!(e, back);
    }

    // -- Appraisal --

    #[test]
    fn test_appraise_pleasant_novel() {
        let dims = AppraisalDimensions {
            novelty: 0.8,
            pleasantness: 0.9,
            goal_conduciveness: 0.7,
            coping_potential: 0.3,
            norm_compatibility: 0.5,
        };
        let a = appraise(&dims).unwrap();
        assert!(a.valence > 0.5); // pleasant
        assert!(a.arousal > 0.3); // novel + low coping = high arousal
    }

    #[test]
    fn test_appraise_neutral() {
        let dims = AppraisalDimensions {
            novelty: 0.0,
            pleasantness: 0.0,
            goal_conduciveness: 0.0,
            coping_potential: 1.0,
            norm_compatibility: 0.0,
        };
        let a = appraise(&dims).unwrap();
        assert!(a.valence.abs() < 1e-10);
        assert!(a.arousal.abs() < 1e-10);
    }

    #[test]
    fn test_appraisal_serde_roundtrip() {
        let dims = AppraisalDimensions {
            novelty: 0.5,
            pleasantness: -0.3,
            goal_conduciveness: 0.2,
            coping_potential: 0.8,
            norm_compatibility: 0.1,
        };
        let json = serde_json::to_string(&dims).unwrap();
        let back: AppraisalDimensions = serde_json::from_str(&json).unwrap();
        assert!((dims.novelty - back.novelty).abs() < 1e-10);
    }

    // -- Regulation --

    #[test]
    fn test_regulate_reduces_intensity() {
        let affect = Affect::new(-0.8, 0.7).unwrap();
        let regulated = regulate(affect, RegulationStrategy::CognitiveChange, 1.0).unwrap();
        assert!(regulated.intensity() < affect.intensity());
    }

    #[test]
    fn test_regulate_zero_effort() {
        let affect = Affect::new(0.5, 0.5).unwrap();
        let regulated = regulate(affect, RegulationStrategy::CognitiveChange, 0.0).unwrap();
        assert!((regulated.valence - affect.valence).abs() < 1e-10);
    }

    #[test]
    fn test_regulation_strategy_ordering() {
        let reappraisal = RegulationStrategy::CognitiveChange.effectiveness();
        let suppression = RegulationStrategy::ResponseModulation.effectiveness();
        assert!(reappraisal > suppression);
    }

    #[test]
    fn test_regulation_strategy_serde_roundtrip() {
        let s = RegulationStrategy::AttentionalDeployment;
        let json = serde_json::to_string(&s).unwrap();
        let back: RegulationStrategy = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    // -- Yerkes-Dodson --

    #[test]
    fn test_yerkes_dodson_optimal() {
        let p = yerkes_dodson(0.5, 0.5, 1.0).unwrap();
        assert!((p - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_yerkes_dodson_inverted_u() {
        let at_opt = yerkes_dodson(0.5, 0.5, 0.5).unwrap();
        let below = yerkes_dodson(0.0, 0.5, 0.5).unwrap();
        let above = yerkes_dodson(1.0, 0.5, 0.5).unwrap();
        assert!(at_opt > below);
        assert!(at_opt > above);
    }

    #[test]
    fn test_yerkes_dodson_floors_at_zero() {
        let p = yerkes_dodson(10.0, 0.5, 0.5).unwrap();
        assert!((p - 0.0).abs() < 1e-10);
    }

    // -- Mood-congruent bias --

    #[test]
    fn test_mood_congruent_bias_same_affect() {
        let affect = Affect::new(0.5, 0.5).unwrap();
        let biased = mood_congruent_bias(0.5, affect, affect, 0.5).unwrap();
        assert!(biased > 0.5); // boost when congruent
    }

    #[test]
    fn test_mood_congruent_bias_opposite_affect() {
        let current = Affect::new(0.8, 0.8).unwrap();
        let memory = Affect::new(-0.8, -0.8).unwrap();
        let same = mood_congruent_bias(0.5, current, current, 0.5).unwrap();
        let opposite = mood_congruent_bias(0.5, current, memory, 0.5).unwrap();
        assert!(same > opposite);
    }

    #[test]
    fn test_mood_congruent_bias_zero_weight() {
        let a = Affect::new(0.5, 0.5).unwrap();
        let b = Affect::new(-0.5, -0.5).unwrap();
        let biased = mood_congruent_bias(0.3, a, b, 0.0).unwrap();
        assert!((biased - 0.3).abs() < 1e-10);
    }
}
