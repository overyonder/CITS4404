#pragma once

#include <vector>

/**
 * Neural Network utilities for neuroevolution
 */

/**
 * Calculate the total number of weights needed for a neural network
 * with the given layer configuration.
 * @param layers Vector of layer sizes (e.g., {8, 16, 4, 1})
 * @return Total number of weights including bias weights
 */
int layers_to_weights(const std::vector<int>& layers);

/**
 * Activation function for neural network neurons.
 * Uses clamped linear activation function (clamps between -1 and 1).
 * @param x Input value
 * @return Activated output value
 */
double activation_function(double x);

/**
 * Evaluate a neural network with given architecture and weights.
 * @param layers Vector defining network layer sizes
 * @param weights Vector of all network weights (must match layers_to_weights size)
 * @param inputs Input values (must match first layer size)
 * @return Output values from the network's output layer
 */
std::vector<double> evaluate_neural_net(
    const std::vector<int>& layers, 
    const std::vector<double>& weights, 
    const std::vector<double>& inputs
); 