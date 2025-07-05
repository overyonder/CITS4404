// IMPORTANT: Add `rand = "0.8"` to your [dependencies] in Cargo.toml
use crate::constants::{INPUT_SIZE, L1_SIZE, L1_WEIGHTS, L2_SIZE, L2_WEIGHTS, TOTAL_WEIGHTS};
use rand::Rng;
use std::{fs, io};

#[derive(Clone, Copy)]
pub struct Individual {
    pub weights: [f32; TOTAL_WEIGHTS],
}

impl Default for Individual {
    fn default() -> Self {
        let mut weights = [0.0; TOTAL_WEIGHTS];
        let mut rng = rand::thread_rng();
        for weight in weights.iter_mut() {
            *weight = rng.gen_range(-1.0..=1.0);
        }
        Self { weights }
    }
}

impl Individual {
    /// Saves the individual's weights to a binary file.
    pub fn save(&self, path: &str) -> io::Result<()> {
        // This is safe because f32 has a fixed memory layout.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                self.weights.as_ptr() as *const u8,
                TOTAL_WEIGHTS * std::mem::size_of::<f32>(),
            )
        };
        fs::write(path, bytes)
    }

    /// Loads an individual's weights from a binary file.
    pub fn load(path: &str) -> io::Result<Self> {
        let bytes = fs::read(path)?;
        if bytes.len() != TOTAL_WEIGHTS * std::mem::size_of::<f32>() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File size does not match expected weight data size.",
            ));
        }

        let mut individual = Individual::default();
        // This is safe due to the size check above.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                individual.weights.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
        }
        Ok(individual)
    }

    pub fn forward(&self, input: &[f32; INPUT_SIZE]) -> f32 {
        let (l1_w, rest) = self.weights.split_at(L1_WEIGHTS);
        let (l2_w, l3_w) = rest.split_at(L2_WEIGHTS);

        let mut a1 = [0.0; L1_SIZE];
        for i in 0..L1_SIZE {
            let offset = i * (INPUT_SIZE + 1);
            let weights = &l1_w[offset..offset + INPUT_SIZE];
            let bias = l1_w[offset + INPUT_SIZE];
            // Apply activation function for the first hidden layer
            a1[i] = (dot(input, weights) + bias).tanh();
        }

        let mut a2 = [0.0; L2_SIZE];
        for i in 0..L2_SIZE {
            let offset = i * (L1_SIZE + 1);
            let weights = &l2_w[offset..offset + L1_SIZE];
            let bias = l2_w[offset + L1_SIZE];
            // Apply activation function for the second hidden layer
            a2[i] = (dot(&a1, weights) + bias).tanh();
        }

        let weights = &l3_w[..L2_SIZE];
        let bias = l3_w[L2_SIZE];
        // Final output layer, tanh activation clamps output to [-1, 1]
        (dot(&a2, weights) + bias).tanh()
    }
}

/// Computes the dot product of two slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
