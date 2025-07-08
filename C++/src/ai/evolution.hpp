#pragma once

#include <vector>

/**
 * Evolutionary Algorithm utilities for neuroevolution
 */

/**
 * Initialize a population of genomes with random values
 * @param genomes Population to randomize (modified in place)
 */
void randomize_genomes(std::vector<std::vector<double>>& genomes);

/**
 * Select the fittest individuals from a population through tournament evaluation
 * @param keep Number of individuals to keep
 * @param population Current population
 * @param layers Neural network layer configuration for evaluation
 * @return Vector of the fittest genomes
 */
std::vector<std::vector<double>> fittest(
    int keep, 
    const std::vector<std::vector<double>>& population,
    const std::vector<int>& layers
);

/**
 * Create offspring through crossover of two parent genomes
 * @param lp Left parent genome
 * @param rp Right parent genome
 * @return Child genome created from crossover
 */
std::vector<double> crossover(
    const std::vector<double>& lp, 
    const std::vector<double>& rp
);

/**
 * Create a mutated copy of a parent genome
 * @param parent Parent genome to mutate
 * @return Mutated copy of the parent
 */
std::vector<double> mutation(const std::vector<double>& parent); 