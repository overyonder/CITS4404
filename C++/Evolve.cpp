#include "Pong.cpp"
#include "NeuralNet.cpp"
#include <cmath>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>
#include <map>
#include <limits>
#include <fstream>

using namespace std;

// Config options
const vector<int> layers = {8, 16, 4, 1};
const int gen_size = 128;
const int gen_limit = 1024;

struct NeuroPlayer : PlayerController {
	vector<int> layers;
	vector<double> weights;
	NeuroPlayer(vector<int> layers, vector<double> weights) : layers(layers), weights(weights) {}
	vector<double> tick(vector<double> state) override {
		return evaluate_neural_net(layers, weights, state);
	}
};

// Given a population of genomes, initalizes them with random values
void randomize_genomes(vector<vector<double>> & genomes) {
	default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
	uniform_real_distribution<double> distribution(-1.0, 1.0);
	auto rng = bind(distribution, generator);
	for (auto & genome : genomes) {
		generate(genome.begin(), genome.end(), rng);
	}
}

// Return the keep fittest genomes from population
vector<vector<double>> fittest(int keep, const vector<vector<double>> & population) {
	map<int, pair<int, int>> scores;
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
	vector<pair<pair<int, int>, int>> rankings;
	for (auto & kv : scores) {
		rankings.push_back(pair<pair<int, int>, int>(kv.second, kv.first));
	}
	// Sort to find keep fittest individuals
	partial_sort(rankings.begin(), rankings.begin() + keep, rankings.end(), greater<pair<pair<int, int>, int>>());
	vector<vector<double>> selected;
	for (int i = 0; i < keep; ++i) {
		selected.push_back(population[rankings[i].second]);
	}
	cerr << " Best score: <" << rankings.front().first.first << ", " << rankings.front().first.second << ">.";
	return selected;
}

// Given two parents, return a mix of them
// Randomized mixing to better suit neural net weights
vector<double> crossover(const vector<double> & lp, const vector<double> & rp) {
	if (lp.size() != rp.size()) throw "crossover: parent genome lengths do not match";
	vector<double> result(lp);
	default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> selector(0, 1);
	for (int i = 0; i < result.size(); ++i) {
		result[i] = (selector(generator) == 1) ? lp[i] : rp[i];
	}
	return result;
}

// Given a parent, return a mutation of it
vector<double> mutation(const vector<double> & parent) {
	default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> selector(0, parent.size() - 1);
	normal_distribution<double> distribution(0.0, 1.0);
	vector<double> result = parent;
	result[selector(generator)] += distribution(generator);
	return result;
}

int main(int argc, char* argv[]) {
    int generations = 100; // Default number of generations
    if (argc > 1) {
        generations = atoi(argv[1]);
    }
    cerr << "Running for " << generations << " generations." << endl;
	// keep survivors + keep^2/2 crossovers + mutations = gen_size;
	const int keep = (int) sqrt(gen_size);

	// Initialize population
	vector<vector<double>> population(gen_size, vector<double>(layers_to_weights(layers)));
	randomize_genomes(population);

	// Log generations for later replay
	ofstream fitlog("fittest.log");
	fitlog << layers.size();
	for (auto l : layers) {
		fitlog << " " << l;
	}
	fitlog << endl;

	// Simple fixed number of generations
	for (int gen = 0; gen < generations; ++gen) {
		cerr << "Evaulating generation " << gen << "...";

		// Take the survivors from the previous generation
		population = fittest(keep, population);

		// Log them
		fitlog << population.size() << endl;
		for (auto & indiv : population) {
			fitlog << indiv.size();
			for (auto & gene : indiv) {
				fitlog << " " << gene;
			}
			fitlog << endl;
		}

		// Add all keep^2/2 crossovers
		for (int i = 0; i < keep; ++i) {
			for (int j = i + 1; j < keep; ++j) {
				population.push_back(crossover(population[i], population[j]));
			}
		}

		// Bolster with randomly selected mutations to achieve gen_size
		default_random_engine generator(chrono::system_clock::now().time_since_epoch().count());
		uniform_int_distribution<int> distribution(0, keep-1);
		while (population.size() < gen_size) {
			population.push_back(mutation(population[distribution(generator)]));
		}

		cerr << " Done." << endl;
	}

	fitlog.close();

	return 0;
}
