//! Item Response Theory — 1PL, 2PL, 3PL models for psychometric measurement.
//!
//! Models the probability of a correct response as a function of person
//! ability (θ) and item parameters (difficulty, discrimination, guessing).

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result, validate_finite, validate_positive};

// ---------------------------------------------------------------------------
// Item Response Models
// ---------------------------------------------------------------------------

/// 1PL (Rasch) model: probability of correct response.
///
/// `P(θ) = 1 / (1 + e^(-(θ - b)))`
///
/// where `θ` is person ability and `b` is item difficulty.
/// All items have equal discrimination (fixed at 1.0).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite.
#[inline]
#[must_use = "returns the probability of correct response without side effects"]
pub fn rasch_probability(ability: f64, difficulty: f64) -> Result<f64> {
    validate_finite(ability, "ability")?;
    validate_finite(difficulty, "difficulty")?;
    Ok(logistic(ability - difficulty))
}

/// 2PL model: probability of correct response with discrimination.
///
/// `P(θ) = 1 / (1 + e^(-a(θ - b)))`
///
/// where `a` is item discrimination (steepness of the ICC),
/// `b` is difficulty, and `θ` is ability.
///
/// Higher `a` → the item discriminates more sharply between
/// high and low ability examinees.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or
/// `discrimination` is non-positive.
#[inline]
#[must_use = "returns the probability of correct response without side effects"]
pub fn two_pl_probability(ability: f64, difficulty: f64, discrimination: f64) -> Result<f64> {
    validate_finite(ability, "ability")?;
    validate_finite(difficulty, "difficulty")?;
    validate_positive(discrimination, "discrimination")?;
    Ok(logistic(discrimination * (ability - difficulty)))
}

/// 3PL model: probability of correct response with guessing.
///
/// `P(θ) = c + (1 - c) / (1 + e^(-a(θ - b)))`
///
/// where `c` is the pseudo-guessing parameter (lower asymptote,
/// typically 1/k for k-choice items), `a` is discrimination,
/// `b` is difficulty, and `θ` is ability.
///
/// Even examinees with very low ability have probability `c` of
/// getting the item correct by guessing.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite,
/// `discrimination` is non-positive, or `guessing` is outside \[0, 1).
#[must_use = "returns the probability of correct response without side effects"]
pub fn three_pl_probability(
    ability: f64,
    difficulty: f64,
    discrimination: f64,
    guessing: f64,
) -> Result<f64> {
    validate_finite(ability, "ability")?;
    validate_finite(difficulty, "difficulty")?;
    validate_positive(discrimination, "discrimination")?;
    validate_finite(guessing, "guessing")?;
    if !(0.0..1.0).contains(&guessing) {
        return Err(BodhError::InvalidParameter(
            "guessing must be in [0, 1)".into(),
        ));
    }
    let p_star = logistic(discrimination * (ability - difficulty));
    Ok(guessing + (1.0 - guessing) * p_star)
}

// ---------------------------------------------------------------------------
// Item Information
// ---------------------------------------------------------------------------

/// Item information function for the 2PL model.
///
/// `I(θ) = a² × P(θ) × (1 - P(θ))`
///
/// Information is maximized at θ = b (the difficulty level) and
/// increases with discrimination. This determines how precisely
/// an item measures ability at a given level.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are non-finite or
/// `discrimination` is non-positive.
#[inline]
#[must_use = "returns the item information without side effects"]
pub fn item_information_2pl(ability: f64, difficulty: f64, discrimination: f64) -> Result<f64> {
    let p = two_pl_probability(ability, difficulty, discrimination)?;
    Ok(discrimination * discrimination * p * (1.0 - p))
}

/// Item information function for the 3PL model.
///
/// `I(θ) = a² × ((P - c) / ((1 - c) × P))² × P × (1 - P)`
///
/// The guessing parameter reduces information, especially at low
/// ability levels where correct responses may be due to guessing
/// rather than knowledge.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid.
#[must_use = "returns the item information without side effects"]
pub fn item_information_3pl(
    ability: f64,
    difficulty: f64,
    discrimination: f64,
    guessing: f64,
) -> Result<f64> {
    let p = three_pl_probability(ability, difficulty, discrimination, guessing)?;
    if p < 1e-15 {
        return Ok(0.0);
    }
    let ratio = (p - guessing) / ((1.0 - guessing) * p);
    Ok(discrimination * discrimination * ratio * ratio * p * (1.0 - p))
}

/// Test information function: sum of item information across items.
///
/// `TI(θ) = Σ I_i(θ)`
///
/// Higher test information → more precise ability estimate at θ.
/// The standard error of the ability estimate is `SE(θ) = 1 / √TI(θ)`.
///
/// Each item is described by `(difficulty, discrimination)`.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any parameter is invalid.
#[must_use = "returns the test information without side effects"]
pub fn test_information_2pl(ability: f64, items: &[(f64, f64)]) -> Result<f64> {
    validate_finite(ability, "ability")?;
    let mut total = 0.0;
    for &(difficulty, discrimination) in items {
        total += item_information_2pl(ability, difficulty, discrimination)?;
    }
    Ok(total)
}

/// Standard error of ability estimate from test information.
///
/// `SE(θ) = 1 / √TI(θ)`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if inputs are invalid or
/// test information is zero.
#[inline]
#[must_use = "returns the standard error without side effects"]
pub fn ability_standard_error(test_info: f64) -> Result<f64> {
    validate_finite(test_info, "test_info")?;
    if test_info <= 0.0 {
        return Err(BodhError::ComputationError(
            "test information must be positive for SE computation".into(),
        ));
    }
    Ok(1.0 / test_info.sqrt())
}

// ---------------------------------------------------------------------------
// Item Parameters
// ---------------------------------------------------------------------------

/// IRT item parameters (3PL model, subsumes 1PL and 2PL).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ItemParameters {
    /// Item difficulty (location on ability scale).
    pub difficulty: f64,
    /// Item discrimination (slope of ICC at inflection point).
    pub discrimination: f64,
    /// Pseudo-guessing parameter (lower asymptote).
    pub guessing: f64,
}

impl ItemParameters {
    /// Probability of correct response at a given ability level.
    ///
    /// # Errors
    ///
    /// Returns [`BodhError::InvalidParameter`] if ability is non-finite.
    #[must_use = "returns the probability without side effects"]
    pub fn probability(&self, ability: f64) -> Result<f64> {
        three_pl_probability(ability, self.difficulty, self.discrimination, self.guessing)
    }

    /// Item information at a given ability level.
    ///
    /// # Errors
    ///
    /// Returns [`BodhError::InvalidParameter`] if ability is non-finite.
    #[must_use = "returns the information without side effects"]
    pub fn information(&self, ability: f64) -> Result<f64> {
        item_information_3pl(ability, self.difficulty, self.discrimination, self.guessing)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Standard logistic function.
#[inline]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Rasch / 1PL --

    #[test]
    fn test_rasch_at_difficulty() {
        // When ability = difficulty, P = 0.5.
        let p = rasch_probability(1.0, 1.0).unwrap();
        assert!((p - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_rasch_high_ability() {
        let p = rasch_probability(3.0, 0.0).unwrap();
        assert!(p > 0.95);
    }

    #[test]
    fn test_rasch_low_ability() {
        let p = rasch_probability(-3.0, 0.0).unwrap();
        assert!(p < 0.05);
    }

    #[test]
    fn test_rasch_monotonic() {
        let p1 = rasch_probability(0.0, 1.0).unwrap();
        let p2 = rasch_probability(1.0, 1.0).unwrap();
        let p3 = rasch_probability(2.0, 1.0).unwrap();
        assert!(p1 < p2);
        assert!(p2 < p3);
    }

    // -- 2PL --

    #[test]
    fn test_2pl_at_difficulty() {
        let p = two_pl_probability(1.0, 1.0, 1.5).unwrap();
        assert!((p - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_2pl_high_discrimination_steeper() {
        // Higher discrimination → steeper ICC → more extreme probabilities.
        let p_low_a = two_pl_probability(1.5, 1.0, 0.5).unwrap();
        let p_high_a = two_pl_probability(1.5, 1.0, 2.0).unwrap();
        // Both above 0.5 (ability > difficulty), but high-a is more extreme.
        assert!(p_high_a > p_low_a);
    }

    #[test]
    fn test_2pl_matches_rasch_at_a1() {
        let rasch = rasch_probability(1.5, 0.5).unwrap();
        let twopl = two_pl_probability(1.5, 0.5, 1.0).unwrap();
        assert!((rasch - twopl).abs() < 1e-10);
    }

    // -- 3PL --

    #[test]
    fn test_3pl_guessing_floor() {
        // Very low ability → P ≈ guessing parameter.
        let p = three_pl_probability(-10.0, 0.0, 1.0, 0.25).unwrap();
        assert!((p - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_3pl_high_ability() {
        // Very high ability → P ≈ 1.0.
        let p = three_pl_probability(10.0, 0.0, 1.0, 0.25).unwrap();
        assert!((p - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_3pl_no_guessing_matches_2pl() {
        let twopl = two_pl_probability(1.0, 0.5, 1.5).unwrap();
        let threepl = three_pl_probability(1.0, 0.5, 1.5, 0.0).unwrap();
        assert!((twopl - threepl).abs() < 1e-10);
    }

    #[test]
    fn test_3pl_invalid_guessing() {
        assert!(three_pl_probability(1.0, 0.0, 1.0, -0.1).is_err());
        assert!(three_pl_probability(1.0, 0.0, 1.0, 1.0).is_err());
    }

    // -- Information --

    #[test]
    fn test_info_2pl_peaks_at_difficulty() {
        let info_at_b = item_information_2pl(1.0, 1.0, 1.5).unwrap();
        let info_away = item_information_2pl(3.0, 1.0, 1.5).unwrap();
        assert!(info_at_b > info_away);
    }

    #[test]
    fn test_info_2pl_known_value() {
        // At θ=b, P=0.5, I = a² × 0.5 × 0.5 = a²/4.
        let info = item_information_2pl(0.0, 0.0, 2.0).unwrap();
        assert!((info - 1.0).abs() < 1e-10); // 4/4 = 1.0
    }

    #[test]
    fn test_info_increases_with_discrimination() {
        let low = item_information_2pl(0.0, 0.0, 0.5).unwrap();
        let high = item_information_2pl(0.0, 0.0, 2.0).unwrap();
        assert!(high > low);
    }

    #[test]
    fn test_info_3pl_less_than_2pl() {
        // Guessing reduces information.
        let info_2pl = item_information_2pl(0.0, 0.0, 1.5).unwrap();
        let info_3pl = item_information_3pl(0.0, 0.0, 1.5, 0.25).unwrap();
        assert!(info_2pl > info_3pl);
    }

    // -- Test information --

    #[test]
    fn test_test_information_additive() {
        let items = vec![(0.0, 1.0), (1.0, 1.0), (-1.0, 1.0)];
        let ti = test_information_2pl(0.0, &items).unwrap();
        let sum: f64 = items
            .iter()
            .map(|&(b, a)| item_information_2pl(0.0, b, a).unwrap())
            .sum();
        assert!((ti - sum).abs() < 1e-10);
    }

    #[test]
    fn test_ability_se() {
        // TI = 4.0 → SE = 1/√4 = 0.5
        let se = ability_standard_error(4.0).unwrap();
        assert!((se - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ability_se_zero_info() {
        assert!(ability_standard_error(0.0).is_err());
    }

    // -- ItemParameters struct --

    #[test]
    fn test_item_params_probability() {
        let item = ItemParameters {
            difficulty: 0.0,
            discrimination: 1.0,
            guessing: 0.25,
        };
        let p = item.probability(-10.0).unwrap();
        assert!((p - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_item_params_information() {
        let item = ItemParameters {
            difficulty: 0.0,
            discrimination: 1.5,
            guessing: 0.0,
        };
        let info = item.information(0.0).unwrap();
        let expected = item_information_3pl(0.0, 0.0, 1.5, 0.0).unwrap();
        assert!((info - expected).abs() < 1e-10);
    }

    // -- Serde roundtrips --

    #[test]
    fn test_item_parameters_serde_roundtrip() {
        let item = ItemParameters {
            difficulty: 1.2,
            discrimination: 0.8,
            guessing: 0.2,
        };
        let json = serde_json::to_string(&item).unwrap();
        let back: ItemParameters = serde_json::from_str(&json).unwrap();
        assert!((item.difficulty - back.difficulty).abs() < 1e-10);
        assert!((item.guessing - back.guessing).abs() < 1e-10);
    }
}
