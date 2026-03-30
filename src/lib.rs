//! # Bodh — Psychology Engine
//!
//! **बोध** (Sanskrit: awareness, understanding)
//!
//! A Rust library for computational psychology: cognition, perception,
//! learning, decision-making, psychometrics, and psychophysics.
//!
//! ## Modules
//!
//! - [`psychophysics`] — Weber-Fechner, Stevens' power law, Fitts' law, Hick's law
//! - [`cognition`] — Working memory, dual process theory, cognitive load
//! - [`learning`] — Ebbinghaus forgetting curve, spaced repetition, conditioning
//! - [`decision`] — Prospect theory, expected utility, bounded rationality
//! - [`perception`] — Signal detection theory (d'), Gestalt principles
//! - [`psychometrics`] — Cronbach's alpha, reliability, Big Five measurement
//! - [`development`] — Piaget stages, Erikson stages
//! - [`emotion`] — Russell circumplex, Ekman basic emotions, appraisal, regulation
//! - [`memory`] — ACT-R base-level activation, spreading activation, retrieval
//! - [`bayesian`] — Bayes' theorem, belief updating, base rate neglect
//! - [`social`] — Asch conformity, social impact, attribution, comparison
//!
//! ## Example
//!
//! ```
//! use bodh::psychophysics;
//!
//! // Fitts' law: index of difficulty for a mouse target
//! let id = psychophysics::fitts_law(256.0, 4.0).unwrap();
//! assert!((id - 7.0).abs() < 1e-10); // 7 bits
//!
//! // Ebbinghaus forgetting curve
//! let retention = bodh::learning::ebbinghaus_forgetting(0.0, 1.0).unwrap();
//! assert!((retention - 1.0).abs() < 1e-10); // perfect at t=0
//! ```

#![warn(missing_docs)]

pub mod bayesian;
pub mod cognition;
pub mod decision;
pub mod development;
pub mod emotion;
pub mod error;
pub mod learning;
pub mod memory;
pub mod perception;
pub mod psychometrics;
pub mod psychophysics;
pub mod social;

pub use error::BodhError;
