//! Utility functions shared across different modules.

use crate::config::Activation;

/// Applies a non-linear activation function to a value.
/// Performs a dot product of two slices of `f32`.
/// Panics if the slices are not of equal length.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn apply_activation(x: f32, activation: Activation) -> f32 {
    match activation {
        Activation::Tanh => x.tanh(),
        Activation::Relu => x.max(0.0),
        Activation::Atan => x.atan(),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Linear => x,
    }
}
