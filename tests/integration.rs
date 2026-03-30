//! Integration tests for bodh.

use bodh::bayesian;
use bodh::decision;
use bodh::emotion;
use bodh::learning;
use bodh::memory;
use bodh::perception;
use bodh::psychophysics;
use bodh::social;

#[test]
fn test_fitts_law_d256_w4() {
    let id = psychophysics::fitts_law(256.0, 4.0).unwrap();
    assert!((id - 7.0).abs() < 1e-10);
}

#[test]
fn test_hicks_law_8_choices_gives_3b() {
    let rt = psychophysics::hicks_law(8, 1.0).unwrap();
    assert!((rt - 3.0).abs() < 1e-10);
}

#[test]
fn test_ebbinghaus_at_zero() {
    let r = learning::ebbinghaus_forgetting(0.0, 1.0).unwrap();
    assert!((r - 1.0).abs() < 1e-10);
}

#[test]
fn test_prospect_theory_loss_aversion_integration() {
    let gain = decision::prospect_theory_value(200.0, 100.0, 0.88, 0.88, 2.25).unwrap();
    let loss = decision::prospect_theory_value(0.0, 100.0, 0.88, 0.88, 2.25).unwrap();
    assert!(gain > 0.0);
    assert!(loss < 0.0);
    assert!(loss.abs() > gain.abs());
}

#[test]
fn test_d_prime_discriminability() {
    let d = perception::d_prime(0.9, 0.1).unwrap();
    assert!(d > 2.0); // good discrimination
}

#[test]
fn test_weber_fechner_no_change() {
    let p = psychophysics::weber_fechner(100.0, 100.0, 1.0).unwrap();
    assert!((p - 0.0).abs() < 1e-10);
}

#[test]
fn test_stevens_power_law_zero_stimulus() {
    let s = psychophysics::stevens_power_law(0.0, 1.0, 0.33).unwrap();
    assert!((s - 0.0).abs() < 1e-10);
}

#[test]
fn test_spaced_repetition_increasing_intervals() {
    let i1 = learning::spaced_repetition_interval(1, 2.5).unwrap();
    let i2 = learning::spaced_repetition_interval(2, 2.5).unwrap();
    let i3 = learning::spaced_repetition_interval(3, 2.5).unwrap();
    assert!(i1 < i2);
    assert!(i2 < i3);
}

// -- v0.2.0 modules --

#[test]
fn test_emotion_appraisal_to_classification() {
    // Appraise a threatening stimulus → should classify as fear-like.
    let dims = emotion::AppraisalDimensions {
        novelty: 0.9,
        pleasantness: -0.8,
        goal_conduciveness: -0.7,
        coping_potential: 0.1,
        norm_compatibility: 0.0,
    };
    let affect = emotion::appraise(&dims).unwrap();
    assert!(affect.valence < 0.0); // unpleasant
    assert!(affect.arousal > 0.0); // high arousal
}

#[test]
fn test_yerkes_dodson_performance_curve() {
    // Complex task: lower optimal arousal, narrow spread.
    let low = emotion::yerkes_dodson(0.2, 0.4, 0.3).unwrap();
    let optimal = emotion::yerkes_dodson(0.4, 0.4, 0.3).unwrap();
    let high = emotion::yerkes_dodson(0.8, 0.4, 0.3).unwrap();
    assert!(optimal > low);
    assert!(optimal > high);
}

#[test]
fn test_actr_retrieval_pipeline() {
    // Full ACT-R pipeline: base-level → spreading → retrieval probability.
    let history = memory::ChunkHistory {
        presentation_ages: vec![1.0, 5.0, 20.0],
    };
    let base = memory::base_level_activation(&history, 0.5).unwrap();
    let total = memory::spreading_activation(base, &[(1.0, 0.3)]).unwrap();
    let prob = memory::retrieval_probability(total, 0.0, 0.4).unwrap();
    assert!(prob > 0.5); // should be retrievable
}

#[test]
fn test_bayesian_medical_diagnosis() {
    // Classic: rare disease (1%), good test (sens=99%, spec=95%).
    // Most positives are false positives.
    let posterior = bayesian::bayes_posterior(0.01, 0.99, 0.05).unwrap();
    assert!(posterior < 0.5); // still more likely healthy
    assert!(posterior > 0.1); // but much higher than prior
}

#[test]
fn test_bayesian_sequential_evidence() {
    // Two positive tests should dramatically increase confidence.
    let after_one = bayesian::sequential_update(0.01, &[(0.99, 0.05)]).unwrap();
    let after_two = bayesian::sequential_update(0.01, &[(0.99, 0.05), (0.99, 0.05)]).unwrap();
    assert!(after_two > after_one);
    assert!(after_two > 0.5);
}

#[test]
fn test_social_conformity_drops_without_unanimity() {
    let unanimous = social::asch_conformity(5, 1.0, 0.37, 0.3).unwrap();
    let dissent = social::asch_conformity(5, 0.8, 0.37, 0.3).unwrap();
    assert!(unanimous > dissent);
}

#[test]
fn test_attribution_with_bias() {
    // External situation without bias → external attribution.
    let info = social::CovariationInfo {
        consensus: 0.9,
        distinctiveness: 0.9,
        consistency: 0.9,
    };
    assert_eq!(
        social::kelley_attribution(&info),
        social::AttributionType::External
    );
    // Same info with strong FAE bias → internal attribution.
    let biased = social::fundamental_attribution_error(&info, 0.8).unwrap();
    assert_eq!(
        social::kelley_attribution(&biased),
        social::AttributionType::Internal
    );
}
