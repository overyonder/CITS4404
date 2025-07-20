use std::hash::{BuildHasher, Hasher, RandomState};

pub fn random_u64() -> u64 {
    RandomState::new().build_hasher().finish()
}

pub fn random_f32() -> f32 {
    random_u64() as f32
}

pub fn random_f32_range(min: f32, max: f32) -> f32 {
    min + (max - min) * random_f32() / f32::MAX
}

pub fn random_usize_range(min: usize, max: usize) -> usize {
    random_f32_range(min as f32, max as f32) as usize
}

pub fn normal_sample(mean: f32, std_dev: f32) -> f32 {
    let u1 = random_f32_range(0., 1.);
    let u2 = random_f32_range(0., 1.);
    // Box-Muller transform
    let z0 = (-2. * u1.ln()).sqrt() * (2. * std::f32::consts::PI * u2).cos();
    mean + z0 * std_dev
}
