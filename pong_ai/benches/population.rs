use criterion::{Criterion, criterion_group, criterion_main};
use pong_ai::train;
use std::hint::black_box;

fn bench_population_train(c: &mut Criterion) {
    c.bench_function("Population::train()", |b| {
        b.iter(|| {
            black_box(train());
        })
    });
}

criterion_group!(benches, bench_population_train);
criterion_main!(benches);
