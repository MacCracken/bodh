//! Error types for the bodh psychology engine.

use core::fmt;
use serde::{Deserialize, Serialize};

/// Errors that can occur in bodh operations.
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[non_exhaustive]
pub enum BodhError {
    /// A parameter was invalid (e.g., negative where positive required).
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// A model computation failed.
    #[error("model failed: {0}")]
    ModelFailed(String),

    /// A measurement or observation was invalid.
    #[error("measurement error: {0}")]
    MeasurementError(String),

    /// A general computation error.
    #[error("computation error: {0}")]
    ComputationError(String),
}

/// Result type alias for bodh operations.
pub type Result<T> = core::result::Result<T, BodhError>;

/// Validate that a value is finite and non-NaN.
#[inline]
pub(crate) fn validate_finite(value: f64, name: &str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(BodhError::InvalidParameter(fmt::format(format_args!(
            "{name} must be finite, got {value}"
        ))))
    }
}

/// Validate that a value is positive (> 0).
#[inline]
pub(crate) fn validate_positive(value: f64, name: &str) -> Result<()> {
    validate_finite(value, name)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(BodhError::InvalidParameter(fmt::format(format_args!(
            "{name} must be positive, got {value}"
        ))))
    }
}

/// Validate that a value is non-negative (>= 0).
#[inline]
pub(crate) fn validate_non_negative(value: f64, name: &str) -> Result<()> {
    validate_finite(value, name)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(BodhError::InvalidParameter(fmt::format(format_args!(
            "{name} must be non-negative, got {value}"
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BodhError::InvalidParameter("test".into());
        assert_eq!(err.to_string(), "invalid parameter: test");
    }

    #[test]
    fn test_error_serde_roundtrip() {
        let err = BodhError::ComputationError("overflow".into());
        let json = serde_json::to_string(&err).unwrap();
        let back: BodhError = serde_json::from_str(&json).unwrap();
        assert_eq!(err.to_string(), back.to_string());
    }

    #[test]
    fn test_validate_finite() {
        assert!(validate_finite(1.0, "x").is_ok());
        assert!(validate_finite(f64::NAN, "x").is_err());
        assert!(validate_finite(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "x").is_ok());
        assert!(validate_positive(0.0, "x").is_err());
        assert!(validate_positive(-1.0, "x").is_err());
    }

    #[test]
    fn test_validate_non_negative() {
        assert!(validate_non_negative(0.0, "x").is_ok());
        assert!(validate_non_negative(1.0, "x").is_ok());
        assert!(validate_non_negative(-0.1, "x").is_err());
    }
}
