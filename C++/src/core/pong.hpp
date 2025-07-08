#pragma once

#include "geometry.hpp"
#include <vector>
#include <random>

/**
 * Abstract interface for pong player controllers
 */
struct PlayerController {
    virtual std::vector<double> tick(std::vector<double> state) = 0;
    virtual ~PlayerController() = default;
};

/**
 * Main Pong game simulation class
 */
class PongGame {
public:
    // Game configuration
    const int tickrate = 60;        // For animation convenience
    int max_score = 1;              // Tracked stats and limit
    
    // Score tracking
    int left_score = 0, right_score = 0;
    int left_returns = 0, right_returns = 0;
    int left_shots = 0, right_shots = 0;
    
    // Game dimensions and physics
    const double length = 400;
    const double width = 300;
    const double paddle_width = width/8;
    const double paddle_max_vel = width/tickrate;
    const Point ball_start_vel = Point(length/tickrate, length/tickrate);
    
    // Game state
    Point ball_pos = Point(0, 0);
    Point ball_vel = ball_start_vel;
    double left_pos = 0, left_vel = 0;
    double right_pos = 0, right_vel = 0;
    
    // Configuration
    bool enable_random = true;
    
    // Constructor
    PongGame(PlayerController& left, PlayerController& right);
    
    // Game simulation
    void tick();
    std::pair<int, int> simulate();

private:
    PlayerController& left;
    PlayerController& right;
    
    // Random number generation for paddle deflection
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;
}; 