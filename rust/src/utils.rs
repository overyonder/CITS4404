//! Utility functions for performance optimization, mathematical operations, and benchmarking.
//!
//! This module provides helper functions used throughout the codebase for:
//! - High-performance mathematical operations (dot products, activation functions)
//! - Educational demonstrations of optimization techniques
//! - Clear examples of Rust performance patterns for students
//!
//! # Teaching Note: Performance Engineering in Evolutionary Algorithms
//! Performance optimization is critical in evolutionary algorithms due to their
//! computational intensity. This module demonstrates several optimization principles:
//! - Iterator chain optimization for mathematical operations
//! - Function dispatching patterns for runtime polymorphism
//! - Clear, readable code that still performs well
//! - Educational examples that students can understand and modify

use crate::config::Activation;

/// High-performance dot product for vector inputs using iterator optimization.
///
/// # Teaching Note: Iterator Optimization in Rust
/// This function demonstrates modern Rust performance patterns:
/// - **Iterator chains**: More readable than manual loops while maintaining performance
/// - **Lazy evaluation**: Operations are fused together by the compiler for efficiency
/// - **SIMD potential**: Rust's iterator patterns enable automatic vectorization
/// - **Memory safety**: Bounds checking at compile-time through slice types
/// - **Zero-cost abstractions**: High-level code compiles to optimized assembly
///
/// # Performance Characteristics:
/// - Time complexity: O(N) where N is vector length
/// - Space complexity: O(1) - operates with iterator state only
/// - Cache efficiency: Sequential access pattern is cache-friendly
/// - Branch prediction: Minimal branching in the hot loop
///
/// # Mathematical Foundation:
/// Computes the standard dot product: Σ(aᵢ × bᵢ) for i = 0 to n-1
/// This is a fundamental operation in neural networks for computing weighted sums.
///
/// # Usage in Neural Networks:
/// ```text
/// neuron_input = dot(input_vector, weight_vector) + bias
/// ```
/// Each neuron computes this dot product to determine its activation level.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vector lengths must match for dot product");
    
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * y)
        .sum()
}

/// Applies the specified activation function to a raw neural network output.
///
/// # Teaching Note: Activation Functions in Neural Networks
/// Activation functions introduce **non-linearity** into neural networks, enabling them
/// to learn complex patterns beyond linear relationships. Without activation functions,
/// even deep networks would be equivalent to simple linear regression.
/// 
/// ## Function Characteristics:
/// - **ClampedLinear**: `f(x) = clamp(x, -1, 1)`
///   - Fast to compute, bounded output range
///   - Used for C++ compatibility in this project
///   - Good for problems where output range is known
/// 
/// - **Tanh**: `f(x) = tanh(x)`  
///   - Range: (-1, 1), smooth S-shaped curve
///   - Zero-centered (mean activation ≈ 0)
///   - Well-behaved gradients for backpropagation (though not used in this EA)
/// 
/// - **ReLU**: `f(x) = max(0, x)`
///   - Most popular in modern deep learning
///   - Fast computation, helps avoid vanishing gradients
///   - Can suffer from "dead neurons" problem
/// 
/// - **Atan**: `f(x) = atan(x)`
///   - Range: (-π/2, π/2) ≈ (-1.57, 1.57)
///   - Similar shape to tanh but different scaling
/// 
/// - **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`
///   - Range: (0, 1), classic S-shaped curve
///   - Can interpret output as probability
///   - Prone to gradient saturation at extremes
/// 
/// - **Linear**: `f(x) = x`
///   - No non-linearity, acts as pass-through
///   - Useful for output layers or debugging
///   - Multiple linear layers collapse to single linear transformation
///
/// # Performance Note:
/// Uses Rust's pattern matching for compile-time function dispatch. The compiler
/// can often inline these simple mathematical operations for optimal performance.
pub fn apply_activation(value: f32, activation: Activation) -> f32 {
    match activation {
        Activation::ClampedLinear => value.clamp(-1.0, 1.0),
        Activation::Tanh => value.tanh(),
        Activation::Relu => value.max(0.0),
        Activation::Atan => value.atan(),
        Activation::Sigmoid => 1.0 / (1.0 + (-value).exp()),
        Activation::Linear => value,
    }
}




