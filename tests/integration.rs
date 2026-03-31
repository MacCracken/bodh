//! Integration tests for bodh.

use bodh::attention;
use bodh::bayesian;
use bodh::cognition;
use bodh::decision;
use bodh::emotion;
use bodh::irt;
use bodh::learning;
use bodh::memory;
use bodh::motivation;
use bodh::perception;
use bodh::psychometrics;
use bodh::psychophysics;
use bodh::social;
use bodh::stress;

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

// -- Motivation --

#[test]
fn test_sdt_needs_drive_motivation() {
    let high = motivation::BasicNeeds {
        autonomy: 0.9,
        competence: 0.9,
        relatedness: 0.9,
    };
    let low = motivation::BasicNeeds {
        autonomy: 0.1,
        competence: 0.1,
        relatedness: 0.1,
    };
    assert!(motivation::predict_motivation(&high) > motivation::predict_motivation(&low));
}

#[test]
fn test_flow_requires_skill_challenge_match() {
    let flow = motivation::flow_state(0.8, 0.8).unwrap();
    let anxiety = motivation::flow_state(0.8, 0.2).unwrap();
    let boredom = motivation::flow_state(0.2, 0.8).unwrap();
    assert!(flow > anxiety);
    assert!(flow > boredom);
}

#[test]
fn test_goal_gradient_accelerates() {
    let early = motivation::goal_gradient(0.1, 1.0, 1.5).unwrap();
    let mid = motivation::goal_gradient(0.5, 1.0, 1.5).unwrap();
    let late = motivation::goal_gradient(0.9, 1.0, 1.5).unwrap();
    assert!(mid > early);
    assert!(late > mid);
}

// -- Attention --

#[test]
fn test_posner_cueing_asymmetry() {
    let benefit = 30.0;
    let cost = 50.0;
    // Cost > benefit is the standard finding.
    assert!(cost > benefit);
    let valid =
        attention::posner_cueing_rt(300.0, attention::CueValidity::Valid, benefit, cost).unwrap();
    let invalid =
        attention::posner_cueing_rt(300.0, attention::CueValidity::Invalid, benefit, cost).unwrap();
    assert!((invalid - valid - (benefit + cost)).abs() < 1e-10);
}

#[test]
fn test_visual_search_feature_vs_conjunction() {
    let feat =
        attention::visual_search_rt(attention::SearchType::Feature, 20, 400.0, 25.0, true).unwrap();
    let conj =
        attention::visual_search_rt(attention::SearchType::Conjunction, 20, 400.0, 25.0, true)
            .unwrap();
    assert!(conj > feat); // conjunction search is slower
}

#[test]
fn test_attentional_blink_recovery() {
    let lag2 = attention::attentional_blink(2, 0.95, 0.4, 3.0, 1.5).unwrap();
    let lag8 = attention::attentional_blink(8, 0.95, 0.4, 3.0, 1.5).unwrap();
    assert!(lag8 > lag2); // recovery from blink
}

// -- IRT --

#[test]
fn test_irt_3pl_bounds() {
    // 3PL: low ability → guessing, high ability → ~1.0.
    let low = irt::three_pl_probability(-5.0, 0.0, 1.0, 0.25).unwrap();
    let high = irt::three_pl_probability(5.0, 0.0, 1.0, 0.25).unwrap();
    assert!((low - 0.25).abs() < 0.01);
    assert!((high - 1.0).abs() < 0.01);
}

#[test]
fn test_irt_information_peaks_at_difficulty() {
    let at_b = irt::item_information_2pl(0.0, 0.0, 1.5).unwrap();
    let away = irt::item_information_2pl(3.0, 0.0, 1.5).unwrap();
    assert!(at_b > away);
}

#[test]
fn test_irt_test_information_additive() {
    let items = vec![(-1.0, 1.0), (0.0, 1.5), (1.0, 1.0)];
    let ti = irt::test_information_2pl(0.0, &items).unwrap();
    assert!(ti > 0.0);
    let se = irt::ability_standard_error(ti).unwrap();
    assert!(se > 0.0);
    assert!(se < 2.0); // 3 items give moderate precision
}

// -- Working memory updating --

#[test]
fn test_nback_degrades_with_level() {
    let a1 = cognition::nback_accuracy(1, 4.0, 0.4).unwrap();
    let a3 = cognition::nback_accuracy(3, 4.0, 0.4).unwrap();
    assert!(a1 > a3);
}

#[test]
fn test_complex_span_demand_reduces_capacity() {
    let full = cognition::complex_span_capacity(4.0, 0.0, 0.8).unwrap();
    let loaded = cognition::complex_span_capacity(4.0, 0.6, 0.8).unwrap();
    assert!(full > loaded);
}

// -- Encoding/retrieval --

#[test]
fn test_levels_of_processing_depth() {
    let shallow = memory::encoding_strength(memory::ProcessingLevel::Structural, 0.0, 1.0).unwrap();
    let deep = memory::encoding_strength(memory::ProcessingLevel::Semantic, 0.0, 1.0).unwrap();
    assert!(deep > shallow);
}

#[test]
fn test_encoding_specificity_context_match() {
    let matched = memory::encoding_specificity(0.8, 1.0, 2.0).unwrap();
    let mismatched = memory::encoding_specificity(0.8, 0.3, 2.0).unwrap();
    assert!(matched > mismatched);
}

// -- Personality --

#[test]
fn test_big_five_scoring_reverse_items() {
    let items = vec![4.0, 2.0, 5.0, 1.0]; // item 1 and 3 reverse-keyed
    let score = psychometrics::score_dimension(&items, &[1, 3], 5.0).unwrap();
    // adjusted: 4, (5+1-2)=4, 5, (5+1-1)=5 → mean = 18/4 = 4.5
    assert!((score - 4.5).abs() < 1e-10);
}

// -- Stress --

#[test]
fn test_stress_transactional_model() {
    let secondary = stress::SecondaryAppraisal {
        perceived_control: 0.2,
        coping_resources: 0.2,
        self_efficacy: 0.2,
    };
    let s = stress::stress_intensity(stress::PrimaryAppraisal::Threat, &secondary).unwrap();
    assert!(s > 0.4); // high threat + low coping = high stress
}

#[test]
fn test_burnout_risk_chronic_stress() {
    let short = stress::burnout_risk(0.7, 1.0, 1.0).unwrap();
    let chronic = stress::burnout_risk(0.7, 10.0, 1.0).unwrap();
    assert!(chronic > short);
    assert!(chronic > 0.5); // substantial risk after prolonged exposure
}
