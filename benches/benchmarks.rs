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

criterion_group!(
    benches,
    bench_weber_fechner,
    bench_fitts_law,
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
);
criterion_main!(benches);
