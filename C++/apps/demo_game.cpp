#include "../src/ai/players/simple_player.hpp"
#include "../src/core/pong.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <algorithm>

int main() {
    SimplePlayer left;
    SimplePlayer right;
    PongGame pong(left, right);
    
    std::cout << "Pong Demo: Simple AI vs Simple AI" << std::endl;
    std::cout << "Press Ctrl+C to exit" << std::endl << std::endl;
    
    while (std::max(pong.left_score, pong.right_score) < pong.max_score) {
        std::cout << " ";
        for (int i = 0; i < static_cast<int>(pong.length) / 10; ++i)
            std::cout << "=";
        std::cout << " " << std::endl;
        
        for (int i = static_cast<int>(-pong.width) / 20; i <= static_cast<int>(pong.width) / 20; ++i) {
            if (std::abs(pong.left_pos/10 - i) <= pong.paddle_width / 20)
                std::cout << "|";
            else
                std::cout << " ";
                
            for (int j = 0; j < static_cast<int>(pong.length) / 10; ++j) {
                if (static_cast<int>(pong.ball_pos.y / 10) == i && 
                    static_cast<int>(pong.ball_pos.x / 10 + pong.length / 20) == j) {
                    std::cout << "O";
                } else {
                    std::cout << " ";
                }
            }
            
            if (std::abs(pong.right_pos/10 - i) <= pong.paddle_width / 20)
                std::cout << "|";
            else
                std::cout << " ";
            std::cout << std::endl;
        }
        
        std::cout << " ";
        for (int i = 0; i < static_cast<int>(pong.length) / 10; ++i)
            std::cout << "=";
        std::cout << " " << std::endl;
        
        std::cout << "ball_pos: " << pong.ball_pos << "\tball_vel: " << pong.ball_vel << std::endl;
        std::cout << "left_pos: " << pong.left_pos << "\tleft_vel: " << pong.left_vel << std::endl;
        std::cout << "right_pos: " << pong.right_pos << "\tright_vel: " << pong.right_vel << std::endl;
        std::cout << "left_score: " << pong.left_score << "\tright_score: " << pong.right_score << std::endl;
        
        pong.tick();
        std::cout.flush();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000/pong.tickrate));
    }
    
    std::cout << "FINAL SCORES: " << pong.left_score << ", " << pong.right_score << std::endl;
    return 0;
} 