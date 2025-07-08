#include "../src/ai/evolution.hpp"
#include "../src/ai/neural_net.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>

// Config options
const std::vector<int> layers = {8, 16, 4, 1};
const int gen_size = 128;

int main(int argc, char* argv[]) {
    int generations = 100; // Default number of generations
    if (argc > 1) {
        generations = std::atoi(argv[1]);
    }
    std::cerr << "Running neuroevolution for " << generations << " generations." << std::endl;
    
    // keep survivors + keep^2/2 crossovers + mutations = gen_size;
    const int keep = static_cast<int>(std::sqrt(gen_size));

    // Initialize population
    std::vector<std::vector<double>> population(gen_size, std::vector<double>(layers_to_weights(layers)));
    randomize_genomes(population);

    // Log generations for later replay
    std::ofstream fitlog("fittest.log");
    fitlog << layers.size();
    for (auto l : layers) {
        fitlog << " " << l;
    }
    fitlog << std::endl;

    // Simple fixed number of generations
    for (int gen = 0; gen < generations; ++gen) {
        std::cerr << "Evaluating generation " << gen << "...";

        // Take the survivors from the previous generation
        population = fittest(keep, population, layers);

        // Log them
        fitlog << population.size() << std::endl;
        for (auto& indiv : population) {
            fitlog << indiv.size();
            for (auto& gene : indiv) {
                fitlog << " " << gene;
            }
            fitlog << std::endl;
        }

        // Add all keep^2/2 crossovers
        for (int i = 0; i < keep; ++i) {
            for (int j = i + 1; j < keep; ++j) {
                population.push_back(crossover(population[i], population[j]));
            }
        }

        // Bolster with randomly selected mutations to achieve gen_size
        std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<int> distribution(0, keep-1);
        while (population.size() < gen_size) {
            population.push_back(mutation(population[distribution(generator)]));
        }

        std::cerr << " Done." << std::endl;
    }

    fitlog.close();
    std::cerr << "Evolution complete. Results saved to fittest.log" << std::endl;

    return 0;
} 