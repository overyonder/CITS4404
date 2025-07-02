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

int main() {
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
	for (int gen = 0; gen < gen_limit; ++gen) {
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

	// When we are done, animate the best player from the last generation playing against itself for the user.
	while (true) {
		cerr << "Press Enter to run simulation" << endl;
		cin.ignore();
		cerr << endl << endl << endl;
		NeuroPlayer left(layers, population[0]);
		NeuroPlayer right(layers, population[0]);
		PongGame pong(left, right);
		pong.max_score = 2;
		while (max(pong.left_score, pong.right_score) < pong.max_score) {
			cout << " ";
			for (int i = 0; i < 2 * (int) pong.length / 10; ++i)
				cout << "=";
			cout << " " << endl;
			for (int i = (int) -pong.width / 20; i <= (int) pong.width / 20; ++i) {
				if (abs((int) pong.left_pos/10 - i) <= (int) pong.paddle_width / 20)
					cout << "|";
				else
					cout << " ";
				for (int j = 0; j < 2 * (int) pong.length / 10; ++j) {
					if ((int) (pong.ball_pos.y / 10) == i && (int) (2 * (int) pong.ball_pos.x / 10 + 2 * (int) pong.length / 20) == j) {
						cout << "O";
					} else {
						cout << " ";
					}
				}
				if (abs((int) pong.right_pos/10 - i) <= (int) pong.paddle_width / 20)
					cout << "|";
				else
					cout << " ";
				cout << endl;
			}
			cout << " ";
			for (int i = 0; i < 2 * (int) pong.length / 10; ++i)
				cout << "=";
			cerr << " " << endl;
			cerr << "ball_pos: " << pong.ball_pos << "\tball_vel: " << pong.ball_vel << endl;
			cerr << "left_pos: " << pong.left_pos << "\tleft_vel: " << pong.left_vel << endl;
			cerr << "right_pos: " << pong.right_pos << "\tright_vel: " << pong.right_vel << endl;
			cerr << "left_score: " << pong.left_score << "\tright_score: " << pong.right_score << endl;
			pong.tick();
			this_thread::sleep_for(chrono::milliseconds(1000/pong.tickrate));
		}
		cerr << "SCORES: " << pong.left_score << ", " << pong.right_score << endl;
	}
	return 0;
}
