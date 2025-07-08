#include "../src/ai/players/neural_player.hpp"
#include "../src/core/pong.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>

// Cross-platform screen clearing
void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

int main(int argc, char* argv[]) {
    std::vector<int> layers;
    std::vector<std::vector<std::vector<double>>> generations;

    // Load generations from log
    std::ifstream fitlog;
    if (argc < 2) {
        fitlog.open("fittest.log");
    } else {
        fitlog.open(argv[1]);
    }

    if (!fitlog.is_open()) {
        std::cerr << "Error: Could not open evolution log file." << std::endl;
        return 1;
    }

    int L;
    fitlog >> L;
    layers = std::vector<int>(L);
    for (int l = 0; l < L; ++l) {
        fitlog >> layers[l];
    }

    fitlog >> std::ws;

    while (fitlog.good()) {
        int I;
        fitlog >> I;
        std::vector<std::vector<double>> generation;
        for (int i = 0; i < I; ++i) {
            int G;
            fitlog >> G;
            std::vector<double> genome;
            for (int g = 0; g < G; ++g) {
                double gene;
                fitlog >> gene;
                genome.push_back(gene);
            }
            generation.push_back(genome);
        }
        generations.push_back(generation);
        fitlog >> std::ws;
    }

    fitlog.close();

    std::cout << generations.size() << " generations loaded" << std::endl;
    std::cout << "Select generation to simulate: ";
    std::cout.flush();

    int gen;
    std::cin >> gen;

    // Repeatedly prompt the user for which generation to animate
    while (0 <= gen && gen < generations.size()) {
        NeuroPlayer left(layers, generations[gen][0]);
        NeuroPlayer right(layers, generations[gen][0]);
        PongGame pong(left, right);
        pong.max_score = 3;
        
        while (std::max(pong.left_score, pong.right_score) < pong.max_score) {
            clear_screen();
            
            // Draw top border
            std::cout << " ";
            for (int i = 0; i < 2 * static_cast<int>(pong.length) / 10; ++i)
                std::cout << "=";
            std::cout << " " << std::endl;
            
            // Draw game field
            for (int i = static_cast<int>(-pong.width) / 20; i <= static_cast<int>(pong.width) / 20; ++i) {
                if (std::abs(static_cast<int>(pong.left_pos)/10 - i) <= static_cast<int>(pong.paddle_width) / 20)
                    std::cout << "|";
                else
                    std::cout << " ";
                    
                for (int j = 0; j < 2 * static_cast<int>(pong.length) / 10; ++j) {
                    if (static_cast<int>(pong.ball_pos.y / 10) == i && 
                        static_cast<int>(2 * static_cast<int>(pong.ball_pos.x) / 10 + 2 * static_cast<int>(pong.length) / 20) == j) {
                        std::cout << "O";
                    } else {
                        std::cout << " ";
                    }
                }
                
                if (std::abs(static_cast<int>(pong.right_pos)/10 - i) <= static_cast<int>(pong.paddle_width) / 20)
                    std::cout << "|";
                else
                    std::cout << " ";
                std::cout << std::endl;
            }
            
            // Draw bottom border
            std::cout << " ";
            for (int i = 0; i < 2 * static_cast<int>(pong.length) / 10; ++i)
                std::cout << "=";
            std::cout << " " << std::endl;
            
            // Draw game info
            std::cout << "ball_pos: " << pong.ball_pos << "\tball_vel: " << pong.ball_vel << std::endl;
            std::cout << "left_pos: " << pong.left_pos << "\tleft_vel: " << pong.left_vel << std::endl;
            std::cout << "right_pos: " << pong.right_pos << "\tright_vel: " << pong.right_vel << std::endl;
            std::cout << "left_score: " << pong.left_score << "\tright_score: " << pong.right_score << std::endl;
            
            pong.tick();
            std::this_thread::sleep_for(std::chrono::milliseconds(1000/pong.tickrate));
        }
        
        std::cout << "SCORES: " << pong.left_score << ", " << pong.right_score << std::endl << std::endl;

        std::cout << "Select generation to simulate: ";
        std::cout.flush();
        std::cin >> gen;
    }

    return 0;
} 