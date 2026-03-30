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
}
