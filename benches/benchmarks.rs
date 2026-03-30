use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_weber_fechner(c: &mut Criterion) {
    c.bench_function("psychophysics/weber_fechner", |b| {
        b.iter(|| bodh::psychophysics::weber_fechner(black_box(200.0), black_box(100.0), black_box(1.0)))
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

criterion_group!(
    benches,
    bench_weber_fechner,
    bench_fitts_law,
    bench_ebbinghaus,
    bench_prospect_theory,
    bench_d_prime,
);
criterion_main!(benches);
