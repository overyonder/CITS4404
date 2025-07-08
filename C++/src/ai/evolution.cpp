#include "evolution.hpp"
#include "players/neural_player.hpp"
#include "../core/pong.hpp"
#include <algorithm>
#include <functional>
#include <random>
#include <map>
#include <chrono>
#include <iostream>
#include <cmath>
#include <stdexcept>

void randomize_genomes(std::vector<std::vector<double>>& genomes) {
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);
    auto rng = std::bind(distribution, generator);
    for (auto& genome : genomes) {
        std::generate(genome.begin(), genome.end(), rng);
    }
}

std::vector<std::vector<double>> fittest(
    int keep, 
    const std::vector<std::vector<double>>& population,
    const std::vector<int>& layers
) {
    std::map<int, std::pair<int, int>> scores;
    
    // Tournament evaluation: every player plays against every other player
    for (int li = 0; li < population.size(); ++li) {
        for (int ri = 0; ri < population.size(); ++ri) {
            if (li == ri) continue;
            
            NeuroPlayer left(layers, population[li]);
            NeuroPlayer right(layers, population[ri]);
            PongGame game(left, right);
            game.simulate();
            
            // Prioritize plays, then wins, as even a bad player can score a lucky win
            scores[li].first += game.left_returns + game.left_shots;
            scores[ri].first += game.right_returns + game.right_shots;
            
            if (game.left_score > game.right_score) {
                ++scores[li].second;
            } else {
                ++scores[ri].second;
            }
        }
    }
    
    // Create rankings based on scores
    std::vector<std::pair<std::pair<int, int>, int>> rankings;
    for (auto& kv : scores) {
        rankings.push_back(std::pair<std::pair<int, int>, int>(kv.second, kv.first));
    }
    
    // Sort to find keep fittest individuals
    std::partial_sort(rankings.begin(), rankings.begin() + keep, rankings.end(), 
                     std::greater<std::pair<std::pair<int, int>, int>>());
    
    std::vector<std::vector<double>> selected;
    for (int i = 0; i < keep; ++i) {
        selected.push_back(population[rankings[i].second]);
    }
    
    std::cerr << " Best score: <" << rankings.front().first.first 
              << ", " << rankings.front().first.second << ">.";
    return selected;
}

std::vector<double> crossover(
    const std::vector<double>& lp, 
    const std::vector<double>& rp
) {
    if (lp.size() != rp.size()) 
        throw std::runtime_error("crossover: parent genome lengths do not match");
        
    std::vector<double> result(lp);
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> selector(0, 1);
    
    for (int i = 0; i < result.size(); ++i) {
        result[i] = (selector(generator) == 1) ? lp[i] : rp[i];
    }
    return result;
}

std::vector<double> mutation(const std::vector<double>& parent) {
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> selector(0, parent.size() - 1);
    std::normal_distribution<double> distribution(0.0, 1.0);
    
    std::vector<double> result = parent;
    result[selector(generator)] += distribution(generator);
    return result;
} 