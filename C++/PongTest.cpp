#include "Pong.cpp"
#include <cmath>
#include <chrono>
#include <thread>

struct SimplePlayer : PlayerController {
	vector<double> tick(vector<double> state) override {
		if (state[1] - state[4] < 0)
			return vector<double>({-1});
		else if (state[1] - state[4] > 0)
			return vector<double>({1});
		else
			return vector<double>({0});
	}
};

int main() {
	SimplePlayer left;
	SimplePlayer right;
	PongGame pong(left, right);
	while (max(pong.left_score, pong.right_score) < pong.max_score) {
		cout << " ";
		for (int i = 0; i < (int) pong.length / 10; ++i)
			cout << "=";
		cout << " " << endl;
		for (int i = (int) -pong.width / 20; i <= (int) pong.width / 20; ++i) {
			if (abs(pong.left_pos/10 - i) <= pong.paddle_width / 20)
				cout << "|";
			else
				cout << " ";
			for (int j = 0; j < (int) pong.length / 10; ++j) {
				if ((int) (pong.ball_pos.y / 10) == i && (int) (pong.ball_pos.x / 10 + pong.length / 20) == j) {
					cout << "O";
				} else {
					cout << " ";
				}
			}
			if (abs(pong.right_pos/10 - i) <= pong.paddle_width / 20)
				cout << "|";
			else
				cout << " ";
			cout << endl;
		}
		cout << " ";
		for (int i = 0; i < (int) pong.length / 10; ++i)
			cout << "=";
		cout << " " << endl;
		cout << "ball_pos: " << pong.ball_pos << "\tball_vel: " << pong.ball_vel << endl;
		cout << "left_pos: " << pong.left_pos << "\tleft_vel: " << pong.left_vel << endl;
		cout << "right_pos: " << pong.right_pos << "\tright_vel: " << pong.right_vel << endl;
		cout << "left_score: " << pong.left_score << "\tright_score: " << pong.right_score << endl;
		pong.tick();
		//cin.ignore();
		this_thread::sleep_for(chrono::milliseconds(1000/pong.tickrate));
	}
	cout << "SCORES: " << pong.left_score << ", " << pong.right_score << endl;
	return 0;
}