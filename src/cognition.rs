//! Cognition — working memory, dual process theory, cognitive load.

use serde::{Deserialize, Serialize};

use crate::error::{Result, validate_non_negative, validate_positive};

/// Working memory model based on Baddeley's multicomponent model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemory {
    /// Phonological loop capacity (number of items, typically 4-7).
    pub phonological_loop_capacity: usize,
    /// Visuospatial sketchpad capacity (number of items, typically 3-4).
    pub visuospatial_capacity: usize,
    /// Central executive load (0.0 = idle, 1.0 = saturated).
    pub central_executive_load: f32,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self {
            phonological_loop_capacity: 7,
            visuospatial_capacity: 4,
            central_executive_load: 0.0,
        }
    }
}

impl WorkingMemory {
    /// Total working memory capacity (Miller's law: 7 +/- 2).
    #[inline]
    #[must_use]
    pub fn total_capacity(&self) -> usize {
        self.phonological_loop_capacity + self.visuospatial_capacity
    }

    /// Available capacity accounting for central executive load.
    #[inline]
    #[must_use]
    pub fn available_capacity(&self) -> f64 {
        let total = self.total_capacity() as f64;
        total * (1.0 - self.central_executive_load as f64)
    }
}

/// Dual process theory (Kahneman System 1 / System 2).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualProcess {
    /// System 1 (fast, intuitive) response time in milliseconds.
    pub system1_speed_ms: f64,
    /// System 2 (slow, deliberative) response time in milliseconds.
    pub system2_speed_ms: f64,
}

impl Default for DualProcess {
    fn default() -> Self {
        Self {
            system1_speed_ms: 200.0,
            system2_speed_ms: 2000.0,
        }
    }
}

impl DualProcess {
    /// Speed ratio: how much faster System 1 is than System 2.
    ///
    /// Returns `f64::INFINITY` if `system1_speed_ms` is zero (infinitely fast).
    #[inline]
    #[must_use]
    pub fn speed_ratio(&self) -> f64 {
        if self.system1_speed_ms == 0.0 {
            return f64::INFINITY;
        }
        self.system2_speed_ms / self.system1_speed_ms
    }
}

/// Compute cognitive load for a set of tasks.
///
/// Each task has an intrinsic load (difficulty) and an extraneous load
/// (presentation complexity). Total cognitive load is the sum.
///
/// Returns a value where 1.0 = at working memory capacity.
///
/// # Errors
///
/// Returns [`crate::BodhError::InvalidParameter`] if any load value is non-finite
/// or capacity is zero.
#[must_use = "returns the cognitive load ratio without side effects"]
pub fn cognitive_load(task_loads: &[(f64, f64)], working_memory_capacity: f64) -> Result<f64> {
    validate_positive(working_memory_capacity, "working_memory_capacity")?;
    let mut total = 0.0;
    for (intrinsic, extraneous) in task_loads {
        validate_non_negative(*intrinsic, "intrinsic load")?;
        validate_non_negative(*extraneous, "extraneous load")?;
        total += intrinsic + extraneous;
    }
    Ok(total / working_memory_capacity)
}

/// Filter stimuli through an attention bottleneck.
///
/// Given a set of stimuli with salience scores, returns the indices of
/// stimuli that pass through the attention filter (up to `capacity` items),
/// sorted by salience (highest first).
///
/// # Errors
///
/// Returns [`crate::BodhError::InvalidParameter`] if any salience is non-finite.
#[must_use = "returns filtered stimuli indices without side effects"]
pub fn attention_bottleneck(salience_scores: &[f64], capacity: usize) -> Result<Vec<usize>> {
    for (i, s) in salience_scores.iter().enumerate() {
        if !s.is_finite() {
            return Err(crate::BodhError::InvalidParameter(format!(
                "salience[{i}] must be finite, got {s}"
            )));
        }
    }
    let mut indexed: Vec<(usize, f64)> = salience_scores.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
    indexed.truncate(capacity);
    Ok(indexed.into_iter().map(|(i, _)| i).collect())
}

// ---------------------------------------------------------------------------
// Working Memory Updating (N-back, Complex Span)
// ---------------------------------------------------------------------------

/// N-back task performance: predicted accuracy as a function of n-level
/// and working memory capacity.
///
/// `accuracy = capacity_factor × e^(-decay × (n - 1))`
///
/// where `capacity_factor = min(capacity / n, 1.0)` reflects the load
/// relative to capacity, and `decay` controls how fast accuracy drops
/// with increasing n (typically 0.3–0.5).
///
/// At n=1, accuracy ≈ capacity_factor (near ceiling for most people).
/// At n=3+, accuracy drops substantially.
///
/// # Errors
///
/// Returns [`crate::BodhError::InvalidParameter`] if n is 0, capacity is
/// non-positive, or decay is non-finite.
#[must_use = "returns the predicted accuracy without side effects"]
pub fn nback_accuracy(n: usize, capacity: f64, decay: f64) -> Result<f64> {
    if n == 0 {
        return Err(crate::BodhError::InvalidParameter(
            "n must be at least 1".into(),
        ));
    }
    validate_positive(capacity, "capacity")?;
    crate::error::validate_finite(decay, "decay")?;

    let capacity_factor = (capacity / n as f64).min(1.0);
    let n_penalty = (-decay * (n as f64 - 1.0)).exp();
    Ok((capacity_factor * n_penalty).clamp(0.0, 1.0))
}

/// Complex span capacity: effective capacity under concurrent
/// processing demands.
///
/// `effective = storage_capacity × (1 − processing_demand) × efficiency`
///
/// where `storage_capacity` is the baseline (typically 3–5 items),
/// `processing_demand` is the fraction of resources used by the
/// processing task (0–1), and `efficiency` reflects individual
/// differences in time-sharing ability (0–1, typical ≈ 0.7).
///
/// # Errors
///
/// Returns [`crate::BodhError::InvalidParameter`] if values are non-finite.
#[inline]
#[must_use = "returns the effective capacity without side effects"]
pub fn complex_span_capacity(
    storage_capacity: f64,
    processing_demand: f64,
    efficiency: f64,
) -> Result<f64> {
    validate_positive(storage_capacity, "storage_capacity")?;
    crate::error::validate_finite(processing_demand, "processing_demand")?;
    crate::error::validate_finite(efficiency, "efficiency")?;
    let demand = processing_demand.clamp(0.0, 1.0);
    let eff = efficiency.clamp(0.0, 1.0);
    Ok(storage_capacity * (1.0 - demand) * eff)
}

/// Working memory updating cost: RT penalty for replacing an item
/// in working memory with a new one.
///
/// `RT = base_rt + switch_cost × n_updates + interference × similarity`
///
/// where `n_updates` is how many items have been updated so far (proactive
/// interference builds), and `similarity` (0–1) between old and new items
/// increases interference.
///
/// # Errors
///
/// Returns [`crate::BodhError::InvalidParameter`] if inputs are non-finite
/// or `base_rt` is non-positive.
#[inline]
#[must_use = "returns the updating RT without side effects"]
pub fn updating_cost(
    base_rt: f64,
    switch_cost: f64,
    n_updates: usize,
    interference: f64,
    similarity: f64,
) -> Result<f64> {
    validate_positive(base_rt, "base_rt")?;
    crate::error::validate_finite(switch_cost, "switch_cost")?;
    crate::error::validate_finite(interference, "interference")?;
    crate::error::validate_finite(similarity, "similarity")?;
    let sim = similarity.clamp(0.0, 1.0);
    Ok(base_rt + switch_cost * n_updates as f64 + interference * sim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_working_memory_default() {
        let wm = WorkingMemory::default();
        assert_eq!(wm.total_capacity(), 11);
        assert!((wm.available_capacity() - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_working_memory_loaded() {
        let wm = WorkingMemory {
            phonological_loop_capacity: 7,
            visuospatial_capacity: 4,
            central_executive_load: 0.5,
        };
        assert!((wm.available_capacity() - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_dual_process_speed_ratio() {
        let dp = DualProcess::default();
        assert!((dp.speed_ratio() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cognitive_load_basic() {
        let tasks = vec![(0.3, 0.1), (0.2, 0.05)];
        let load = cognitive_load(&tasks, 1.0).unwrap();
        assert!((load - 0.65).abs() < 1e-10);
    }

    #[test]
    fn test_cognitive_load_overload() {
        let tasks = vec![(0.8, 0.5)];
        let load = cognitive_load(&tasks, 1.0).unwrap();
        assert!(load > 1.0); // overloaded
    }

    #[test]
    fn test_attention_bottleneck() {
        let salience = vec![0.2, 0.9, 0.5, 0.1, 0.7];
        let filtered = attention_bottleneck(&salience, 3).unwrap();
        assert_eq!(filtered.len(), 3);
        assert_eq!(filtered[0], 1); // highest salience
        assert_eq!(filtered[1], 4);
        assert_eq!(filtered[2], 2);
    }

    #[test]
    fn test_attention_bottleneck_capacity_exceeds() {
        let salience = vec![0.5, 0.3];
        let filtered = attention_bottleneck(&salience, 10).unwrap();
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_working_memory_serde_roundtrip() {
        let wm = WorkingMemory::default();
        let json = serde_json::to_string(&wm).unwrap();
        let back: WorkingMemory = serde_json::from_str(&json).unwrap();
        assert_eq!(
            wm.phonological_loop_capacity,
            back.phonological_loop_capacity
        );
    }

    #[test]
    fn test_dual_process_speed_ratio_zero_system1() {
        let dp = DualProcess {
            system1_speed_ms: 0.0,
            system2_speed_ms: 2000.0,
        };
        assert!(dp.speed_ratio().is_infinite());
    }

    #[test]
    fn test_cognitive_load_zero_capacity() {
        let tasks = vec![(0.3, 0.1)];
        assert!(cognitive_load(&tasks, 0.0).is_err());
    }

    #[test]
    fn test_cognitive_load_empty_tasks() {
        let tasks: Vec<(f64, f64)> = vec![];
        let load = cognitive_load(&tasks, 1.0).unwrap();
        assert!((load - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_attention_bottleneck_nan_salience() {
        let salience = vec![0.5, f64::NAN, 0.3];
        assert!(attention_bottleneck(&salience, 2).is_err());
    }

    #[test]
    fn test_dual_process_serde_roundtrip() {
        let dp = DualProcess::default();
        let json = serde_json::to_string(&dp).unwrap();
        let back: DualProcess = serde_json::from_str(&json).unwrap();
        assert!((dp.system1_speed_ms - back.system1_speed_ms).abs() < 1e-10);
    }

    // -- Working memory updating --

    #[test]
    fn test_nback_1back_near_ceiling() {
        let acc = nback_accuracy(1, 4.0, 0.4).unwrap();
        assert!(acc > 0.9);
    }

    #[test]
    fn test_nback_decreases_with_n() {
        let a1 = nback_accuracy(1, 4.0, 0.4).unwrap();
        let a2 = nback_accuracy(2, 4.0, 0.4).unwrap();
        let a3 = nback_accuracy(3, 4.0, 0.4).unwrap();
        assert!(a1 > a2);
        assert!(a2 > a3);
    }

    #[test]
    fn test_nback_higher_capacity_better() {
        // Use n=5 so capacity differences matter (cap/n < 1.0 for low cap).
        let low = nback_accuracy(5, 3.0, 0.4).unwrap();
        let high = nback_accuracy(5, 7.0, 0.4).unwrap();
        assert!(high > low);
    }

    #[test]
    fn test_nback_zero_n() {
        assert!(nback_accuracy(0, 4.0, 0.4).is_err());
    }

    #[test]
    fn test_complex_span_no_demand() {
        let cap = complex_span_capacity(4.0, 0.0, 1.0).unwrap();
        assert!((cap - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_span_full_demand() {
        let cap = complex_span_capacity(4.0, 1.0, 1.0).unwrap();
        assert!(cap.abs() < 1e-10);
    }

    #[test]
    fn test_complex_span_typical() {
        // 4 items, 50% processing, 70% efficiency → 4 × 0.5 × 0.7 = 1.4
        let cap = complex_span_capacity(4.0, 0.5, 0.7).unwrap();
        assert!((cap - 1.4).abs() < 1e-10);
    }

    #[test]
    fn test_updating_cost_no_updates() {
        let rt = updating_cost(300.0, 50.0, 0, 100.0, 0.5).unwrap();
        // base + 0 switches + 100*0.5 = 350
        assert!((rt - 350.0).abs() < 1e-10);
    }

    #[test]
    fn test_updating_cost_increases() {
        let rt0 = updating_cost(300.0, 50.0, 0, 100.0, 0.5).unwrap();
        let rt3 = updating_cost(300.0, 50.0, 3, 100.0, 0.5).unwrap();
        assert!(rt3 > rt0);
    }
}
