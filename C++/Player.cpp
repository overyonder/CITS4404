#include "Pong.cpp"
#include "NeuralNet.cpp"
#include <cmath>
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <fstream>

using namespace std;

struct NeuroPlayer : PlayerController {
	vector<int> layers;
	vector<double> weights;
	NeuroPlayer(vector<int> layers, vector<double> weights) : layers(layers), weights(weights) {}
	vector<double> tick(vector<double> state) override {
		return evaluate_neural_net(layers, weights, state);
	}
};

int main(int argc, char * argv[]) {
	vector<int> layers;
	vector<vector<vector<double>>> generations;

	// Load generations from log
	ifstream fitlog;
	if (argc < 2) {
		fitlog.open("fittest.log");
	} else {
		fitlog.open(argv[1]);
	}

	int L;
	fitlog >> L;
	layers = vector<int>(L);
	for (int l = 0; l < L; ++l) {
		fitlog >> layers[l];
	}

	fitlog >> ws;

	while (fitlog.good()) {
		int I;
		fitlog >> I;
		vector<vector<double>> generation;
		for (int i = 0; i < I; ++i) {
			int G;
			fitlog >> G;
			vector<double> genome;
			for (int g = 0; g < G; ++g) {
				double gene;
				fitlog >> gene;
				genome.push_back(gene);
			}
			generation.push_back(genome);
		}
		generations.push_back(generation);
		fitlog >> ws;
	}

	fitlog.close();

	cout << generations.size() << " generations loaded" << endl;
	cout << "Select generation to simulate: ";
	cout.flush();

	int gen;
	cin >> gen;

	// Repeatedly prompt the user for which generation to animate
	while (0 <= gen && gen < generations.size()) {
		NeuroPlayer left(layers, generations[gen][0]);
		NeuroPlayer right(layers, generations[gen][0]);
		PongGame pong(left, right);
		pong.max_score = 3;
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
			cout << " " << endl;
			cout << "ball_pos: " << pong.ball_pos << "\tball_vel: " << pong.ball_vel << endl;
			cout << "left_pos: " << pong.left_pos << "\tleft_vel: " << pong.left_vel << endl;
			cout << "right_pos: " << pong.right_pos << "\tright_vel: " << pong.right_vel << endl;
			cout << "left_score: " << pong.left_score << "\tright_score: " << pong.right_score << endl;
			pong.tick();
			cout.flush();
			this_thread::sleep_for(chrono::milliseconds(1000/pong.tickrate));
		}
		cout << "SCORES: " << pong.left_score << ", " << pong.right_score << endl << endl;

		cout << "Select generation to simulate: ";
		cout.flush();
		cin >> gen;
	}

	return 0;
}