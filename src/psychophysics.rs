//! Psychophysics — relationships between physical stimuli and perception.
//!
//! Implements Weber-Fechner law, Stevens' power law, Fitts' law, and Hick's law.

use serde::{Deserialize, Serialize};

use crate::error::{validate_finite, validate_positive, BodhError, Result};

/// Weber-Fechner law: perceived intensity as logarithmic function of stimulus.
///
/// `p = k * ln(S / S0)`
///
/// where `S` is stimulus intensity, `S0` is reference intensity, and `k` is
/// a modality-specific constant.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if any input is non-positive or non-finite.
#[inline]
#[must_use = "returns the perceived intensity without side effects"]
pub fn weber_fechner(stimulus_intensity: f64, reference: f64, k: f64) -> Result<f64> {
    validate_positive(stimulus_intensity, "stimulus_intensity")?;
    validate_positive(reference, "reference")?;
    validate_finite(k, "k")?;
    Ok(k * (stimulus_intensity / reference).ln())
}

/// Weber fraction: the just-noticeable difference (JND) ratio.
///
/// `w = delta_I / I`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if reference intensity is non-positive.
#[inline]
#[must_use = "returns the Weber fraction without side effects"]
pub fn weber_fraction(delta_intensity: f64, reference_intensity: f64) -> Result<f64> {
    validate_finite(delta_intensity, "delta_intensity")?;
    validate_positive(reference_intensity, "reference_intensity")?;
    Ok(delta_intensity / reference_intensity)
}

/// Stevens' power law: sensation magnitude as a power function of stimulus.
///
/// `psi = k * S^n`
///
/// where `S` is stimulus magnitude, `k` is a scaling constant, and `n` is
/// a modality-specific exponent (e.g., brightness ~0.33, loudness ~0.67,
/// electric shock ~3.5).
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if stimulus is negative or inputs are non-finite.
#[inline]
#[must_use = "returns the sensation magnitude without side effects"]
pub fn stevens_power_law(stimulus: f64, k: f64, exponent: f64) -> Result<f64> {
    crate::error::validate_non_negative(stimulus, "stimulus")?;
    validate_finite(k, "k")?;
    validate_finite(exponent, "exponent")?;
    Ok(k * stimulus.powf(exponent))
}

/// Stevens' power law exponents for common modalities.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum StevensExponent {
    /// Brightness (visual luminance), exponent ~0.33.
    Brightness,
    /// Loudness (sound pressure), exponent ~0.67.
    Loudness,
    /// Vibration (on fingertip), exponent ~0.95.
    Vibration,
    /// Electric shock, exponent ~3.5.
    ElectricShock,
    /// Heaviness (lifted weight), exponent ~1.45.
    Heaviness,
    /// Temperature (warmth), exponent ~1.6.
    Temperature,
}

impl StevensExponent {
    /// Returns the canonical exponent value for this modality.
    #[inline]
    #[must_use]
    pub fn value(self) -> f64 {
        match self {
            Self::Brightness => 0.33,
            Self::Loudness => 0.67,
            Self::Vibration => 0.95,
            Self::ElectricShock => 3.5,
            Self::Heaviness => 1.45,
            Self::Temperature => 1.6,
        }
    }
}

/// Fitts' law: time to move to a target as a function of distance and width.
///
/// `MT = a + b * log2(2D / W)`
///
/// where `D` is distance to target center, `W` is target width, `a` is
/// intercept (reaction time), and `b` is the slope (1/throughput).
///
/// Returns the index of difficulty (in bits) when `a=0, b=1`:
/// `ID = log2(2D / W)`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if distance or width is non-positive.
#[inline]
#[must_use = "returns the movement time without side effects"]
pub fn fitts_law(distance: f64, width: f64) -> Result<f64> {
    validate_positive(distance, "distance")?;
    validate_positive(width, "width")?;
    Ok((2.0 * distance / width).log2())
}

/// Fitts' law with custom intercept and slope.
///
/// `MT = a + b * log2(2D / W)`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if distance or width is non-positive,
/// or if `a` or `b` is non-finite.
#[inline]
#[must_use = "returns the movement time without side effects"]
pub fn fitts_law_full(distance: f64, width: f64, a: f64, b: f64) -> Result<f64> {
    let id = fitts_law(distance, width)?;
    validate_finite(a, "a")?;
    validate_finite(b, "b")?;
    Ok(a + b * id)
}

/// Hick's law (Hick-Hyman law): decision time as a function of number of choices.
///
/// `RT = a + b * log2(n)`
///
/// where `n` is the number of equally probable choices, `a` is the base
/// reaction time, and `b` is the processing time per bit.
///
/// With `a=0`, returns just the information-theoretic component `b * log2(n)`.
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if `choices` is 0 or `b` is non-finite.
#[inline]
#[must_use = "returns the decision time without side effects"]
pub fn hicks_law(choices: usize, b: f64) -> Result<f64> {
    if choices == 0 {
        return Err(BodhError::InvalidParameter(
            "choices must be at least 1".into(),
        ));
    }
    validate_finite(b, "b")?;
    Ok(b * (choices as f64).log2())
}

/// Hick's law with custom intercept.
///
/// `RT = a + b * log2(n)`
///
/// # Errors
///
/// Returns [`BodhError::InvalidParameter`] if `choices` is 0 or parameters are non-finite.
#[inline]
#[must_use = "returns the decision time without side effects"]
pub fn hicks_law_full(choices: usize, a: f64, b: f64) -> Result<f64> {
    validate_finite(a, "a")?;
    let info_component = hicks_law(choices, b)?;
    Ok(a + info_component)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weber_fechner_basic() {
        // When stimulus equals reference, perceived difference is 0.
        let p = weber_fechner(100.0, 100.0, 1.0).unwrap();
        assert!((p - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_weber_fechner_k1_double() {
        // Doubling stimulus with k=1 gives ln(2) ≈ 0.693.
        let p = weber_fechner(200.0, 100.0, 1.0).unwrap();
        assert!((p - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_weber_fechner_invalid() {
        assert!(weber_fechner(0.0, 100.0, 1.0).is_err());
        assert!(weber_fechner(100.0, 0.0, 1.0).is_err());
    }

    #[test]
    fn test_weber_fraction() {
        // 10% JND
        let w = weber_fraction(10.0, 100.0).unwrap();
        assert!((w - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_stevens_power_law_brightness() {
        // Brightness exponent ~0.33. Doubling stimulus should less than double sensation.
        let s1 = stevens_power_law(100.0, 1.0, 0.33).unwrap();
        let s2 = stevens_power_law(200.0, 1.0, 0.33).unwrap();
        assert!(s2 > s1);
        assert!(s2 < 2.0 * s1); // compressive
    }

    #[test]
    fn test_stevens_power_law_linear() {
        // Exponent 1.0 should be linear: psi = k * S.
        let s = stevens_power_law(50.0, 2.0, 1.0).unwrap();
        assert!((s - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_stevens_exponent_values() {
        assert!((StevensExponent::Brightness.value() - 0.33).abs() < 1e-10);
        assert!((StevensExponent::Loudness.value() - 0.67).abs() < 1e-10);
        assert!((StevensExponent::ElectricShock.value() - 3.5).abs() < 1e-10);
    }

    #[test]
    fn test_fitts_law_basic() {
        // D=256, W=4 → ID = log2(2*256/4) = log2(128) = 7.0 bits.
        let id = fitts_law(256.0, 4.0).unwrap();
        assert!((id - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_fitts_law_easy_target() {
        // D=10, W=10 → ID = log2(2*10/10) = log2(2) = 1.0 bit.
        let id = fitts_law(10.0, 10.0).unwrap();
        assert!((id - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fitts_law_invalid() {
        assert!(fitts_law(0.0, 4.0).is_err());
        assert!(fitts_law(256.0, 0.0).is_err());
    }

    #[test]
    fn test_hicks_law_basic() {
        // 8 choices with b=1: RT = 1 * log2(8) = 3.0.
        let rt = hicks_law(8, 1.0).unwrap();
        assert!((rt - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_hicks_law_with_intercept() {
        // 8 choices, a=0.2, b=0.1: RT = 0.2 + 0.1 * 3.0 = 0.5.
        let rt = hicks_law_full(8, 0.2, 0.1).unwrap();
        assert!((rt - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hicks_law_single_choice() {
        // 1 choice: log2(1) = 0, so RT = b*0 = 0.
        let rt = hicks_law(1, 1.0).unwrap();
        assert!((rt - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_hicks_law_zero_choices() {
        assert!(hicks_law(0, 1.0).is_err());
    }

    #[test]
    fn test_stevens_exponent_serde_roundtrip() {
        let exp = StevensExponent::Loudness;
        let json = serde_json::to_string(&exp).unwrap();
        let back: StevensExponent = serde_json::from_str(&json).unwrap();
        assert_eq!(exp, back);
    }
}
