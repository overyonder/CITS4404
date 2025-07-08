#include "neural_net.hpp"
#include <algorithm>
#include <stdexcept>

int layers_to_weights(const std::vector<int>& layers) {
    int total = 0;
    for(int i = 1; i < layers.size(); ++i)
        total += (layers[i-1] + 1) * layers[i];
    return total;
}

double activation_function(double x) {
    if (x < -1.0) return -1.0;
    else if (x > 1.0) return 1.0;
    else return x;
}

std::vector<double> evaluate_neural_net(
    const std::vector<int>& layers, 
    const std::vector<double>& weights, 
    const std::vector<double>& inputs
) {
    if (weights.size() != layers_to_weights(layers)) 
        throw std::runtime_error("evaluate_neural_net: layers and weights do not match");
    if (inputs.size() != layers.front()) 
        throw std::runtime_error("evaluate_neural_net: layers and input size do not match");
        
    // Initialize previous layer as input layer values
    std::vector<double> prev = inputs;
    // Weights are stored serially, use a common iterator
    auto weight = weights.begin();
    
    // For every layer after the input layer up to and including the output layer
    for (auto layer = ++layers.begin(); layer != layers.end(); ++layer) {
        // Calculate weighted input for each neuron in this layer
        std::vector<double> curr(*layer);
        for (double & neuron : curr) {
            for (double value : prev)
                neuron += *(weight++) * value;
            neuron += *(weight++);	// Unity weight (bias)
        }
        // Apply activation function to curr and overwrite prev
        prev.resize(curr.size());
        std::transform(curr.begin(), curr.end(), prev.begin(), activation_function);
    }
    
    // prev currently holds the value of the output neurons, so return it
    return prev;
} 