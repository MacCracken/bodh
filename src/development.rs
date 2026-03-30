//! Developmental psychology — stage theories, cognitive development.

use serde::{Deserialize, Serialize};

/// Piaget's stages of cognitive development.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum PiagetStage {
    /// Birth to ~2 years: sensory experiences + motor actions.
    Sensorimotor,
    /// ~2 to ~7 years: symbolic thinking, egocentrism.
    Preoperational,
    /// ~7 to ~11 years: logical thought about concrete events.
    ConcreteOperational,
    /// ~11+ years: abstract and hypothetical reasoning.
    FormalOperational,
}

impl PiagetStage {
    /// Typical age range for this stage (min, max) in years.
    #[inline]
    #[must_use]
    pub fn typical_age_range(self) -> (f64, f64) {
        match self {
            Self::Sensorimotor => (0.0, 2.0),
            Self::Preoperational => (2.0, 7.0),
            Self::ConcreteOperational => (7.0, 11.0),
            Self::FormalOperational => (11.0, 18.0),
        }
    }

    /// Determine the expected Piaget stage for a given age.
    #[must_use]
    pub fn from_age(age: f64) -> Option<Self> {
        if age < 0.0 || !age.is_finite() {
            return None;
        }
        if age < 2.0 {
            Some(Self::Sensorimotor)
        } else if age < 7.0 {
            Some(Self::Preoperational)
        } else if age < 11.0 {
            Some(Self::ConcreteOperational)
        } else {
            Some(Self::FormalOperational)
        }
    }
}

/// Erikson's psychosocial stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum EriksonStage {
    /// Infancy (0-1.5 years): Trust vs. Mistrust.
    TrustVsMistrust,
    /// Early childhood (1.5-3): Autonomy vs. Shame/Doubt.
    AutonomyVsShame,
    /// Play age (3-5): Initiative vs. Guilt.
    InitiativeVsGuilt,
    /// School age (5-12): Industry vs. Inferiority.
    IndustryVsInferiority,
    /// Adolescence (12-18): Identity vs. Role Confusion.
    IdentityVsRoleConfusion,
    /// Young adulthood (18-40): Intimacy vs. Isolation.
    IntimacyVsIsolation,
    /// Middle adulthood (40-65): Generativity vs. Stagnation.
    GenerativityVsStagnation,
    /// Late adulthood (65+): Integrity vs. Despair.
    IntegrityVsDespair,
}

impl EriksonStage {
    /// Typical age range for this stage (min, max) in years.
    #[inline]
    #[must_use]
    pub fn typical_age_range(self) -> (f64, f64) {
        match self {
            Self::TrustVsMistrust => (0.0, 1.5),
            Self::AutonomyVsShame => (1.5, 3.0),
            Self::InitiativeVsGuilt => (3.0, 5.0),
            Self::IndustryVsInferiority => (5.0, 12.0),
            Self::IdentityVsRoleConfusion => (12.0, 18.0),
            Self::IntimacyVsIsolation => (18.0, 40.0),
            Self::GenerativityVsStagnation => (40.0, 65.0),
            Self::IntegrityVsDespair => (65.0, 100.0),
        }
    }

    /// Determine the expected Erikson stage for a given age.
    #[must_use]
    pub fn from_age(age: f64) -> Option<Self> {
        if age < 0.0 || !age.is_finite() {
            return None;
        }
        if age < 1.5 {
            Some(Self::TrustVsMistrust)
        } else if age < 3.0 {
            Some(Self::AutonomyVsShame)
        } else if age < 5.0 {
            Some(Self::InitiativeVsGuilt)
        } else if age < 12.0 {
            Some(Self::IndustryVsInferiority)
        } else if age < 18.0 {
            Some(Self::IdentityVsRoleConfusion)
        } else if age < 40.0 {
            Some(Self::IntimacyVsIsolation)
        } else if age < 65.0 {
            Some(Self::GenerativityVsStagnation)
        } else {
            Some(Self::IntegrityVsDespair)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_piaget_from_age() {
        assert_eq!(PiagetStage::from_age(0.5), Some(PiagetStage::Sensorimotor));
        assert_eq!(
            PiagetStage::from_age(4.0),
            Some(PiagetStage::Preoperational)
        );
        assert_eq!(
            PiagetStage::from_age(9.0),
            Some(PiagetStage::ConcreteOperational)
        );
        assert_eq!(
            PiagetStage::from_age(15.0),
            Some(PiagetStage::FormalOperational)
        );
    }

    #[test]
    fn test_piaget_invalid_age() {
        assert_eq!(PiagetStage::from_age(-1.0), None);
        assert_eq!(PiagetStage::from_age(f64::NAN), None);
    }

    #[test]
    fn test_piaget_age_range() {
        let (min, max) = PiagetStage::Sensorimotor.typical_age_range();
        assert!((min - 0.0).abs() < 1e-10);
        assert!((max - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_erikson_from_age() {
        assert_eq!(
            EriksonStage::from_age(0.5),
            Some(EriksonStage::TrustVsMistrust)
        );
        assert_eq!(
            EriksonStage::from_age(25.0),
            Some(EriksonStage::IntimacyVsIsolation)
        );
        assert_eq!(
            EriksonStage::from_age(70.0),
            Some(EriksonStage::IntegrityVsDespair)
        );
    }

    #[test]
    fn test_erikson_invalid_age() {
        assert_eq!(EriksonStage::from_age(-1.0), None);
    }

    #[test]
    fn test_piaget_serde_roundtrip() {
        let stage = PiagetStage::ConcreteOperational;
        let json = serde_json::to_string(&stage).unwrap();
        let back: PiagetStage = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, back);
    }

    #[test]
    fn test_erikson_serde_roundtrip() {
        let stage = EriksonStage::IdentityVsRoleConfusion;
        let json = serde_json::to_string(&stage).unwrap();
        let back: EriksonStage = serde_json::from_str(&json).unwrap();
        assert_eq!(stage, back);
    }
}
