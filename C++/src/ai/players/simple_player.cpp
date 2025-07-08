#include "simple_player.hpp"

std::vector<double> SimplePlayer::tick(std::vector<double> state) {
    // State format: [ball_x, ball_y, ball_vel_x, ball_vel_y, paddle_y, paddle_vel_y, opponent_y, opponent_vel_y]
    // Compare ball Y position (state[1]) to paddle Y position (state[4])
    if (state[1] - state[4] < 0)
        return std::vector<double>({-1}); // Move up
    else if (state[1] - state[4] > 0)
        return std::vector<double>({1});  // Move down
    else
        return std::vector<double>({0});  // Stay in place
} 