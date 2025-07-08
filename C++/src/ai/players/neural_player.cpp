#include "neural_player.hpp"

NeuroPlayer::NeuroPlayer(const std::vector<int>& layers, const std::vector<double>& weights) 
    : layers(layers), weights(weights) {
}

std::vector<double> NeuroPlayer::tick(std::vector<double> state) {
    return evaluate_neural_net(layers, weights, state);
} 