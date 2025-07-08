#pragma once

#include "../../core/pong.hpp"
#include "../neural_net.hpp"
#include <vector>

/**
 * Neural Network-based Pong player
 * Uses evolved neural network weights to make playing decisions
 */
class NeuroPlayer : public PlayerController {
public:
    /**
     * Constructor
     * @param layers Neural network layer configuration (e.g., {8, 16, 4, 1})
     * @param weights Network weights (must match layers_to_weights(layers) size)
     */
    NeuroPlayer(const std::vector<int>& layers, const std::vector<double>& weights);
    
    /**
     * Make a decision based on current game state
     * @param state Game state vector (ball pos, ball vel, paddle positions, etc.)
     * @return Vector containing desired paddle velocity (-1 to 1)
     */
    std::vector<double> tick(std::vector<double> state) override;

private:
    std::vector<int> layers;
    std::vector<double> weights;
}; 