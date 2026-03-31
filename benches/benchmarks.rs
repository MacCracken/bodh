use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_weber_fechner(c: &mut Criterion) {
    c.bench_function("psychophysics/weber_fechner", |b| {
        b.iter(|| {
            bodh::psychophysics::weber_fechner(black_box(200.0), black_box(100.0), black_box(1.0))
        })
    });
}

fn bench_fitts_law(c: &mut Criterion) {
    c.bench_function("psychophysics/fitts_law", |b| {
        b.iter(|| bodh::psychophysics::fitts_law(black_box(256.0), black_box(4.0)))
    });
}

fn bench_fitts_law_shannon(c: &mut Criterion) {
    c.bench_function("psychophysics/fitts_law_shannon", |b| {
        b.iter(|| bodh::psychophysics::fitts_law_shannon(black_box(256.0), black_box(4.0)))
    });
}

fn bench_ebbinghaus(c: &mut Criterion) {
    c.bench_function("learning/ebbinghaus_forgetting", |b| {
        b.iter(|| bodh::learning::ebbinghaus_forgetting(black_box(1.0), black_box(2.0)))
    });
}

fn bench_prospect_theory(c: &mut Criterion) {
    c.bench_function("decision/prospect_theory_value", |b| {
        b.iter(|| {
            bodh::decision::prospect_theory_value(
                black_box(200.0),
                black_box(100.0),
                black_box(0.88),
                black_box(0.88),
                black_box(2.25),
            )
        })
    });
}

fn bench_d_prime(c: &mut Criterion) {
    c.bench_function("perception/d_prime", |b| {
        b.iter(|| bodh::perception::d_prime(black_box(0.9), black_box(0.1)))
    });
}

fn bench_stevens_power_law(c: &mut Criterion) {
    c.bench_function("psychophysics/stevens_power_law", |b| {
        b.iter(|| {
            bodh::psychophysics::stevens_power_law(
                black_box(100.0),
                black_box(1.0),
                black_box(0.33),
            )
        })
    });
}

fn bench_hicks_law(c: &mut Criterion) {
    c.bench_function("psychophysics/hicks_law", |b| {
        b.iter(|| bodh::psychophysics::hicks_law(black_box(8), black_box(1.0)))
    });
}

fn bench_cognitive_load(c: &mut Criterion) {
    let tasks = vec![(0.3, 0.1), (0.2, 0.05), (0.4, 0.2), (0.1, 0.05)];
    c.bench_function("cognition/cognitive_load", |b| {
        b.iter(|| bodh::cognition::cognitive_load(black_box(&tasks), black_box(1.0)))
    });
}

fn bench_attention_bottleneck(c: &mut Criterion) {
    let salience = vec![0.2, 0.9, 0.5, 0.1, 0.7, 0.3, 0.8, 0.6];
    c.bench_function("cognition/attention_bottleneck", |b| {
        b.iter(|| bodh::cognition::attention_bottleneck(black_box(&salience), black_box(3)))
    });
}

fn bench_rescorla_wagner(c: &mut Criterion) {
    c.bench_function("learning/rescorla_wagner", |b| {
        b.iter(|| {
            bodh::learning::rescorla_wagner(
                black_box(0.5),
                black_box(0.5),
                black_box(1.0),
                black_box(0.3),
            )
        })
    });
}

fn bench_probability_weighting(c: &mut Criterion) {
    c.bench_function("decision/probability_weighting", |b| {
        b.iter(|| bodh::decision::probability_weighting(black_box(0.3), black_box(0.61)))
    });
}

fn bench_cronbachs_alpha(c: &mut Criterion) {
    let items = vec![
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 5.0],
        vec![1.1, 2.1, 3.1, 4.1, 5.1, 3.1, 2.1, 4.1, 1.1, 5.1],
        vec![0.9, 1.9, 2.9, 3.9, 4.9, 2.9, 1.9, 3.9, 0.9, 4.9],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 5.0],
        vec![1.2, 2.2, 3.2, 4.2, 5.2, 3.2, 2.2, 4.2, 1.2, 5.2],
    ];
    c.bench_function("psychometrics/cronbachs_alpha", |b| {
        b.iter(|| bodh::psychometrics::cronbachs_alpha(black_box(&items)))
    });
}

fn bench_criterion_c(c: &mut Criterion) {
    c.bench_function("perception/criterion_c", |b| {
        b.iter(|| bodh::perception::criterion_c(black_box(0.9), black_box(0.1)))
    });
}

fn bench_yerkes_dodson(c: &mut Criterion) {
    c.bench_function("emotion/yerkes_dodson", |b| {
        b.iter(|| bodh::emotion::yerkes_dodson(black_box(0.7), black_box(0.5), black_box(0.4)))
    });
}

fn bench_appraise(c: &mut Criterion) {
    let dims = bodh::emotion::AppraisalDimensions {
        novelty: 0.8,
        pleasantness: 0.6,
        goal_conduciveness: 0.4,
        coping_potential: 0.3,
        norm_compatibility: 0.5,
    };
    c.bench_function("emotion/appraise", |b| {
        b.iter(|| bodh::emotion::appraise(black_box(&dims)))
    });
}

fn bench_base_level_activation(c: &mut Criterion) {
    let history = bodh::memory::ChunkHistory {
        presentation_ages: vec![1.0, 5.0, 20.0, 60.0, 300.0],
    };
    c.bench_function("memory/base_level_activation", |b| {
        b.iter(|| bodh::memory::base_level_activation(black_box(&history), black_box(0.5)))
    });
}

fn bench_retrieval_probability(c: &mut Criterion) {
    c.bench_function("memory/retrieval_probability", |b| {
        b.iter(|| {
            bodh::memory::retrieval_probability(black_box(1.5), black_box(0.0), black_box(0.4))
        })
    });
}

fn bench_bayes_posterior(c: &mut Criterion) {
    c.bench_function("bayesian/bayes_posterior", |b| {
        b.iter(|| {
            bodh::bayesian::bayes_posterior(black_box(0.01), black_box(0.99), black_box(0.05))
        })
    });
}

fn bench_sequential_update(c: &mut Criterion) {
    let evidence = vec![(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)];
    c.bench_function("bayesian/sequential_update", |b| {
        b.iter(|| bodh::bayesian::sequential_update(black_box(0.5), black_box(&evidence)))
    });
}

fn bench_asch_conformity(c: &mut Criterion) {
    c.bench_function("social/asch_conformity", |b| {
        b.iter(|| {
            bodh::social::asch_conformity(
                black_box(5),
                black_box(1.0),
                black_box(0.37),
                black_box(0.3),
            )
        })
    });
}

fn bench_social_impact(c: &mut Criterion) {
    c.bench_function("social/social_impact", |b| {
        b.iter(|| bodh::social::social_impact(black_box(10.0), black_box(5), black_box(0.5)))
    });
}

fn bench_flow_state(c: &mut Criterion) {
    c.bench_function("motivation/flow_state", |b| {
        b.iter(|| bodh::motivation::flow_state(black_box(0.7), black_box(0.8)))
    });
}

fn bench_goal_gradient(c: &mut Criterion) {
    c.bench_function("motivation/goal_gradient", |b| {
        b.iter(|| bodh::motivation::goal_gradient(black_box(0.7), black_box(1.0), black_box(1.5)))
    });
}

fn bench_posner_cueing(c: &mut Criterion) {
    c.bench_function("attention/posner_cueing_rt", |b| {
        b.iter(|| {
            bodh::attention::posner_cueing_rt(
                black_box(300.0),
                black_box(bodh::attention::CueValidity::Valid),
                black_box(30.0),
                black_box(50.0),
            )
        })
    });
}

fn bench_visual_search(c: &mut Criterion) {
    c.bench_function("attention/visual_search_rt", |b| {
        b.iter(|| {
            bodh::attention::visual_search_rt(
                black_box(bodh::attention::SearchType::Conjunction),
                black_box(20),
                black_box(400.0),
                black_box(25.0),
                black_box(true),
            )
        })
    });
}

fn bench_attentional_blink(c: &mut Criterion) {
    c.bench_function("attention/attentional_blink", |b| {
        b.iter(|| {
            bodh::attention::attentional_blink(
                black_box(3),
                black_box(0.95),
                black_box(0.4),
                black_box(3.0),
                black_box(1.5),
            )
        })
    });
}

fn bench_rasch(c: &mut Criterion) {
    c.bench_function("irt/rasch_probability", |b| {
        b.iter(|| bodh::irt::rasch_probability(black_box(1.0), black_box(0.5)))
    });
}

fn bench_3pl(c: &mut Criterion) {
    c.bench_function("irt/three_pl_probability", |b| {
        b.iter(|| {
            bodh::irt::three_pl_probability(
                black_box(1.0),
                black_box(0.5),
                black_box(1.5),
                black_box(0.25),
            )
        })
    });
}

fn bench_item_info(c: &mut Criterion) {
    c.bench_function("irt/item_information_2pl", |b| {
        b.iter(|| bodh::irt::item_information_2pl(black_box(0.0), black_box(0.0), black_box(1.5)))
    });
}

fn bench_test_info(c: &mut Criterion) {
    let items = vec![(-1.0, 1.0), (0.0, 1.5), (1.0, 1.0), (0.5, 2.0), (-0.5, 1.2)];
    c.bench_function("irt/test_information_2pl", |b| {
        b.iter(|| bodh::irt::test_information_2pl(black_box(0.0), black_box(&items)))
    });
}

fn bench_nback(c: &mut Criterion) {
    c.bench_function("cognition/nback_accuracy", |b| {
        b.iter(|| bodh::cognition::nback_accuracy(black_box(3), black_box(4.0), black_box(0.4)))
    });
}

fn bench_encoding_strength(c: &mut Criterion) {
    c.bench_function("memory/encoding_strength", |b| {
        b.iter(|| {
            bodh::memory::encoding_strength(
                black_box(bodh::memory::ProcessingLevel::Semantic),
                black_box(1.5),
                black_box(0.8),
            )
        })
    });
}

fn bench_stress_intensity(c: &mut Criterion) {
    let secondary = bodh::stress::SecondaryAppraisal {
        perceived_control: 0.3,
        coping_resources: 0.4,
        self_efficacy: 0.5,
    };
    c.bench_function("stress/stress_intensity", |b| {
        b.iter(|| {
            bodh::stress::stress_intensity(
                black_box(bodh::stress::PrimaryAppraisal::Threat),
                black_box(&secondary),
            )
        })
    });
}

fn bench_burnout_risk(c: &mut Criterion) {
    c.bench_function("stress/burnout_risk", |b| {
        b.iter(|| bodh::stress::burnout_risk(black_box(0.7), black_box(5.0), black_box(1.0)))
    });
}

fn bench_profile_distance(c: &mut Criterion) {
    let a = bodh::psychometrics::BigFiveProfile {
        openness: 3.5,
        conscientiousness: 4.0,
        extraversion: 2.5,
        agreeableness: 3.8,
        neuroticism: 2.0,
    };
    let b = bodh::psychometrics::BigFiveProfile {
        openness: 4.0,
        conscientiousness: 3.0,
        extraversion: 3.5,
        agreeableness: 2.8,
        neuroticism: 3.0,
    };
    c.bench_function("psychometrics/profile_distance", |b_iter| {
        b_iter.iter(|| bodh::psychometrics::profile_distance(black_box(&a), black_box(&b)))
    });
}

criterion_group!(
    benches,
    bench_weber_fechner,
    bench_fitts_law,
    bench_fitts_law_shannon,
    bench_ebbinghaus,
    bench_prospect_theory,
    bench_d_prime,
    bench_stevens_power_law,
    bench_hicks_law,
    bench_cognitive_load,
    bench_attention_bottleneck,
    bench_rescorla_wagner,
    bench_probability_weighting,
    bench_cronbachs_alpha,
    bench_criterion_c,
    bench_yerkes_dodson,
    bench_appraise,
    bench_base_level_activation,
    bench_retrieval_probability,
    bench_bayes_posterior,
    bench_sequential_update,
    bench_asch_conformity,
    bench_social_impact,
    bench_flow_state,
    bench_goal_gradient,
    bench_posner_cueing,
    bench_visual_search,
    bench_attentional_blink,
    bench_rasch,
    bench_3pl,
    bench_item_info,
    bench_test_info,
    bench_nback,
    bench_encoding_strength,
    bench_stress_intensity,
    bench_burnout_risk,
    bench_profile_distance,
);
criterion_main!(benches);
