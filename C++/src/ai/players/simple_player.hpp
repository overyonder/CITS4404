#pragma once

#include "../../core/pong.hpp"
#include <vector>

/**
 * Simple ball-following Pong player
 * Makes decisions based purely on ball position relative to paddle
 */
class SimplePlayer : public PlayerController {
public:
    /**
     * Make a decision based on current game state
     * Simply follows the ball by comparing ball Y position to paddle Y position
     * @param state Game state vector (ball pos, ball vel, paddle positions, etc.)
     * @return Vector containing desired paddle velocity (-1, 0, or 1)
     */
    std::vector<double> tick(std::vector<double> state) override;
}; 