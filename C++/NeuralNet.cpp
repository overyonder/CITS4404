#include <vector>
#include <algorithm>

using namespace std;

// Given a vector specifying the layer widths of the neural net, returns the total number of edge weights
int layers_to_weights(const vector<int>& layers) {
	int total = 0;
	for(int i = 1; i < layers.size(); ++i)
		total += (layers[i-1] + 1) * layers[i];
	return total;
}

// Activation function used for each neuron
double activation_function(double x) {
	if (x < -1.0) return -1.0;
	else if (x > 1.0) return 1.0;
	else return x;
}

vector<double> evaluate_neural_net(const vector<int>& layers, const vector<double>& weights, const vector<double>& inputs) {
	if (weights.size() != layers_to_weights(layers)) throw "evaluate_neural_net: layers and weights do not match";
	if (inputs.size() != layers.front()) throw "evaluate_neural_net: layers and input size do not match";
	// Initialize previous layer as input layer values
	vector<double> prev = inputs;
	// Weights are stored serially, use a common iterator
	auto weight = weights.begin();
	// For every layer after the input layer up to and including the output layer
	for (auto layer = ++layers.begin(); layer != layers.end(); ++layer) {
		// Calculate weighted input for each neuron in this layer
		vector<double> curr(*layer);
		for (double & neuron : curr) {
			for (double value : prev)
				neuron += *(weight++) * value;
			neuron += *(weight++);	// Unity weight
		}
		// Apply activation function to curr and overwrite prev
		prev.resize(curr.size());
		transform(curr.begin(), curr.end(), prev.begin(), activation_function);
	}
	// prev currently holds the value of the output neurons, so return it
	return prev;
}