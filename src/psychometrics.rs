//! Psychometrics — measurement of psychological constructs.
//!
//! Reliability, validity, Big Five measurement, Likert scales.

use serde::{Deserialize, Serialize};

use crate::error::{BodhError, Result};

/// An item response with score and confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemResponse {
    /// Score for this item (e.g., 1-5 on Likert scale).
    pub score: f32,
    /// Confidence in the response (0.0-1.0).
    pub confidence: f32,
}

/// Big Five personality dimension being measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub enum BigFiveDimension {
    /// Openness to experience.
    Openness,
    /// Conscientiousness.
    Conscientiousness,
    /// Extraversion.
    Extraversion,
    /// Agreeableness.
    Agreeableness,
    /// Neuroticism.
    Neuroticism,
}

/// Cronbach's alpha: internal consistency reliability.
///
/// `alpha = (k / (k-1)) * (1 - sum(var_i) / var_total)`
///
/// where `k` is the number of items, `var_i` is the variance of each item,
/// and `var_total` is the variance of total scores.
///
/// Alpha >= 0.7 is generally considered acceptable, >= 0.8 good.
///
/// # Errors
///
/// Returns [`BodhError::MeasurementError`] if fewer than 2 items or 2 respondents,
/// or if total variance is zero.
#[must_use = "returns the reliability coefficient without side effects"]
pub fn cronbachs_alpha(items: &[Vec<f32>]) -> Result<f64> {
    let k = items.len();
    if k < 2 {
        return Err(BodhError::MeasurementError(
            "need at least 2 items for Cronbach's alpha".into(),
        ));
    }

    let n = items[0].len();
    if n < 2 {
        return Err(BodhError::MeasurementError(
            "need at least 2 respondents".into(),
        ));
    }

    for item in items {
        if item.len() != n {
            return Err(BodhError::MeasurementError(
                "all items must have the same number of responses".into(),
            ));
        }
    }

    // Compute variance of each item
    let mut sum_item_var = 0.0_f64;
    for item in items {
        sum_item_var += variance_f32(item);
    }

    // Compute total scores for each respondent
    let mut totals = vec![0.0_f64; n];
    for item in items {
        for (j, &score) in item.iter().enumerate() {
            totals[j] += score as f64;
        }
    }

    let var_total = variance_f64(&totals);
    if var_total < f64::EPSILON {
        return Err(BodhError::MeasurementError(
            "total score variance is zero (no variability)".into(),
        ));
    }

    let k_f = k as f64;
    Ok((k_f / (k_f - 1.0)) * (1.0 - sum_item_var / var_total))
}

/// Split-half reliability: correlation between two halves of a test.
///
/// Uses the Spearman-Brown prophecy formula to correct for test length:
/// `r_full = 2 * r_half / (1 + r_half)`
///
/// # Errors
///
/// Returns [`BodhError::MeasurementError`] if fewer than 4 items.
#[must_use = "returns the split-half reliability without side effects"]
pub fn split_half_reliability(items: &[Vec<f32>]) -> Result<f64> {
    if items.len() < 4 {
        return Err(BodhError::MeasurementError(
            "need at least 4 items for split-half reliability".into(),
        ));
    }

    let n = items[0].len();
    for item in items {
        if item.len() != n {
            return Err(BodhError::MeasurementError(
                "all items must have the same number of responses".into(),
            ));
        }
    }

    // Split into odd/even items
    let mut half1 = vec![0.0_f64; n];
    let mut half2 = vec![0.0_f64; n];

    for (i, item) in items.iter().enumerate() {
        let target = if i % 2 == 0 { &mut half1 } else { &mut half2 };
        for (j, &score) in item.iter().enumerate() {
            target[j] += score as f64;
        }
    }

    let r_half = pearson_correlation_f64(&half1, &half2);

    // Spearman-Brown correction
    if (1.0 + r_half).abs() < f64::EPSILON {
        return Err(BodhError::MeasurementError(
            "negative perfect correlation".into(),
        ));
    }
    Ok(2.0 * r_half / (1.0 + r_half))
}

/// Likert scale midpoint.
#[inline]
#[must_use = "returns the midpoint without side effects"]
pub fn likert_midpoint(min: u32, max: u32) -> f64 {
    (min as f64 + max as f64) / 2.0
}

fn variance_f32(data: &[f32]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let ss: f64 = data.iter().map(|&x| (x as f64 - mean).powi(2)).sum();
    ss / (n - 1.0) // sample variance
}

fn variance_f64(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / n;
    let ss: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
    ss / (n - 1.0)
}

fn pearson_correlation_f64(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < f64::EPSILON {
        return 0.0;
    }
    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cronbachs_alpha_high_reliability() {
        // Items that correlate highly should produce high alpha.
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1],
            vec![0.9, 1.9, 2.9, 3.9, 4.9],
        ];
        let alpha = cronbachs_alpha(&items).unwrap();
        assert!(alpha > 0.95);
    }

    #[test]
    fn test_cronbachs_alpha_low_reliability() {
        // Uncorrelated items should produce low alpha.
        let items = vec![
            vec![5.0, 1.0, 3.0, 2.0, 4.0],
            vec![1.0, 5.0, 2.0, 4.0, 3.0],
            vec![3.0, 2.0, 5.0, 1.0, 4.0],
        ];
        let alpha = cronbachs_alpha(&items).unwrap();
        assert!(alpha < 0.5);
    }

    #[test]
    fn test_cronbachs_alpha_zero_variance() {
        // All identical scores → zero total variance → error.
        let items = vec![vec![3.0, 3.0, 3.0], vec![3.0, 3.0, 3.0]];
        assert!(cronbachs_alpha(&items).is_err());
    }

    #[test]
    fn test_cronbachs_alpha_too_few_items() {
        let items = vec![vec![1.0, 2.0, 3.0]];
        assert!(cronbachs_alpha(&items).is_err());
    }

    #[test]
    fn test_split_half_reliability() {
        let items = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1],
            vec![0.9, 1.9, 2.9, 3.9, 4.9],
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ];
        let r = split_half_reliability(&items).unwrap();
        assert!(r > 0.9);
    }

    #[test]
    fn test_likert_midpoint() {
        assert!((likert_midpoint(1, 5) - 3.0).abs() < 1e-10);
        assert!((likert_midpoint(1, 7) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_big_five_serde_roundtrip() {
        let dim = BigFiveDimension::Extraversion;
        let json = serde_json::to_string(&dim).unwrap();
        let back: BigFiveDimension = serde_json::from_str(&json).unwrap();
        assert_eq!(dim, back);
    }

    #[test]
    fn test_item_response_serde_roundtrip() {
        let ir = ItemResponse {
            score: 4.0,
            confidence: 0.8,
        };
        let json = serde_json::to_string(&ir).unwrap();
        let back: ItemResponse = serde_json::from_str(&json).unwrap();
        assert!((ir.score - back.score).abs() < 1e-5);
    }
}
