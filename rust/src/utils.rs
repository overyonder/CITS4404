//! Utility functions shared across different modules.
//!
//! This module provides common mathematical operations used throughout the neural network
//! implementations. These functions are designed to be efficient and well-documented for
//! educational purposes.

use crate::config::Activation;

/// Computes the dot product of two slices of f32 values.
///
/// # Mathematical Definition
/// The dot product of vectors a and b is: dot(a,b) = Σ(aᵢ × bᵢ) for i = 0 to n-1
///
/// # Neural Network Context
/// In neural networks, dot products are fundamental for computing weighted sums:
/// - Each neuron computes: output = Σ(inputᵢ × weightᵢ) + bias
/// - This is exactly a dot product between the input vector and weight vector
///
/// # Performance Notes
/// - Uses iterator chaining for optimal performance and readability
/// - The compiler can often auto-vectorize this loop for SIMD performance
/// - For very large vectors, consider specialized BLAS libraries like `ndarray` or `candle`
///
/// # Panics
/// Panics if the input slices are not of equal length, as the dot product is undefined
/// for vectors of different dimensions.
///
/// # Teaching Note: Compiler Optimizations
/// This simple implementation often compiles to highly optimized SIMD instructions
/// on modern processors. The Rust compiler can recognize this pattern and automatically
/// vectorize it. For educational purposes, this shows how high-level functional code
/// can achieve performance comparable to hand-optimized loops.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), 
        "Dot product requires vectors of equal length: {} vs {}", a.len(), b.len());
    
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Applies a non-linear activation function to a scalar value.
///
/// # Activation Functions in Neural Networks
/// Activation functions introduce non-linearity into neural networks, enabling them to
/// learn complex patterns. Without activation functions, multiple layers would collapse
/// into a single linear transformation, severely limiting the network's expressive power.
///
/// # Available Activation Functions:
///
/// - **ClampedLinear**: `f(x) = clamp(x, -1, 1)` - Linear with bounds, prevents extreme values
/// - **Tanh**: `f(x) = tanh(x)` - S-shaped curve, outputs in (-1, 1), zero-centered
/// - **ReLU**: `f(x) = max(0, x)` - Simple, fast, addresses vanishing gradient problem
/// - **Atan**: `f(x) = atan(x)` - S-shaped, bounded output in (-π/2, π/2)
/// - **Sigmoid**: `f(x) = 1/(1 + e^(-x))` - Classic S-curve, outputs in (0, 1)
/// - **Linear**: `f(x) = x` - No transformation, used in output layers for regression
///
/// # Teaching Note: Activation Function Properties
/// Each activation function has trade-offs:
/// - **ReLU**: Fast, prevents vanishing gradients, but can "die" (always output 0)
/// - **Tanh**: Zero-centered (good for hidden layers), but can saturate
/// - **Sigmoid**: Historically popular, but suffers from vanishing gradients
/// - **Linear**: Only use in output layer; multiple linear layers = single linear layer
///
/// # Performance Optimization
/// The match statement compiles to a jump table, making function selection very fast.
/// Each activation function uses optimized implementations from the standard library.
#[inline]
pub fn apply_activation(x: f32, activation: Activation) -> f32 {
    match activation {
        Activation::ClampedLinear => x.clamp(-1.0, 1.0),
        Activation::Tanh => x.tanh(),
        Activation::Relu => x.max(0.0),
        Activation::Atan => x.atan(),
        Activation::Sigmoid => {
            // Numerically stable sigmoid implementation
            // Avoids overflow for large negative x values
            if x < 0.0 {
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        },
        Activation::Linear => x,
    }
}

/// Performs efficient dot product computation optimized for small, fixed-size arrays.
///
/// # Teaching Note: Specialization for Performance
/// For the small, fixed-size vectors common in this neural network (8 inputs, 16 hidden units),
/// this specialized version can be faster than the generic `dot` function. The compiler
/// can fully unroll these loops and optimize memory access patterns.
///
/// # When to Use
/// Use this for performance-critical code paths with small, known vector sizes.
/// For larger or variable-sized vectors, use the generic `dot` function.
#[inline]
pub fn dot_small<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    let mut sum = 0.0;
    for i in 0..N {
        sum += a[i] * b[i];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot(&a, &b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    }

    #[test]
    fn test_dot_small() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let result = dot_small(&a, &b);
        assert_eq!(result, 32.0);
    }

    #[test]
    #[should_panic(expected = "Dot product requires vectors of equal length")]
    fn test_dot_product_different_lengths() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0, 5.0];
        dot(&a, &b);
    }

    #[test]
    fn test_activation_functions() {
        // Test basic functionality of each activation
        assert_eq!(apply_activation(0.0, Activation::Linear), 0.0);
        assert_eq!(apply_activation(2.0, Activation::ClampedLinear), 1.0);
        assert_eq!(apply_activation(-2.0, Activation::ClampedLinear), -1.0);
        assert_eq!(apply_activation(0.0, Activation::Relu), 0.0);
        assert_eq!(apply_activation(-1.0, Activation::Relu), 0.0);
        assert_eq!(apply_activation(1.0, Activation::Relu), 1.0);
        
        // Test sigmoid bounds
        let sigmoid_result = apply_activation(0.0, Activation::Sigmoid);
        assert!((sigmoid_result - 0.5).abs() < f32::EPSILON);
        
        // Test tanh bounds
        let tanh_result = apply_activation(0.0, Activation::Tanh);
        assert!((tanh_result - 0.0).abs() < f32::EPSILON);
    }
}
