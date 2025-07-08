#include "../src/ai/players/neural_player.hpp"
#include "../src/core/pong.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>
#include <algorithm>
#include <string>
#include <csignal>

// Global flag for clean shutdown
volatile bool running = true;

// Signal handler for clean shutdown
void signal_handler(int signal) {
    running = false;
}

// Cross-platform screen clearing
void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

// Terminal display class for better rendering
class TerminalDisplay {
private:
    int width, height;
    std::vector<std::string> buffer;
    
public:
    TerminalDisplay(int w, int h) : width(w), height(h) {
        buffer.resize(height, std::string(width, ' '));
    }
    
    void clear() {
        for (auto& line : buffer) {
            std::fill(line.begin(), line.end(), ' ');
        }
    }
    
    void set_char(int x, int y, char c) {
        if (x >= 0 && x < width && y >= 0 && y < height) {
            buffer[y][x] = c;
        }
    }
    
    void draw_line(int x1, int y1, int x2, int y2, char c) {
        int dx = abs(x2 - x1);
        int dy = abs(y2 - y1);
        int sx = (x1 < x2) ? 1 : -1;
        int sy = (y1 < y2) ? 1 : -1;
        int err = dx - dy;
        
        int x = x1, y = y1;
        while (true) {
            set_char(x, y, c);
            if (x == x2 && y == y2) break;
            int e2 = 2 * err;
            if (e2 > -dy) {
                err -= dy;
                x += sx;
            }
            if (e2 < dx) {
                err += dx;
                y += sy;
            }
        }
    }
    
    void render() {
        clear_screen();
        for (const auto& line : buffer) {
            std::cout << line << std::endl;
        }
    }
};

int main(int argc, char* argv[]) {
    // Set up signal handler for clean shutdown
    signal(SIGINT, signal_handler);
    
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

    // Create display buffer
    const int display_width = 80;
    const int display_height = 25;
    TerminalDisplay display(display_width, display_height);

    // Repeatedly prompt the user for which generation to animate
    while (running && 0 <= gen && gen < static_cast<int>(generations.size())) {
        NeuroPlayer left(layers, generations[gen][0]);
        NeuroPlayer right(layers, generations[gen][0]);
        PongGame pong(left, right);
        pong.max_score = 3;
        
        while (running && std::max(pong.left_score, pong.right_score) < pong.max_score) {
            display.clear();
            
            // Calculate proper scaling to maintain aspect ratio
            // Game aspect ratio: length/width = 400/300 = 1.33 (wider than tall)
            // Display aspect ratio: width/height = 80/25 = 3.2
            // We need to fit the game within the display while maintaining proportions
            
            double game_aspect = static_cast<double>(pong.length) / pong.width;  // 1.33
            double display_aspect = static_cast<double>(display_width) / display_height;  // 3.2
            
            double scale_x, scale_y;
            int game_width, game_height;
            int offset_x, offset_y;
            
            // Since display is much wider than game aspect ratio, fit to height
            game_height = display_height - 4;  // Leave 2 chars margin on top/bottom
            scale_y = static_cast<double>(game_height) / pong.width;
            game_width = static_cast<int>(pong.length * scale_y);
            scale_x = scale_y;  // Maintain aspect ratio
            
            offset_x = (display_width - game_width) / 2;
            offset_y = 2;
            
            // Draw top and bottom borders
            for (int x = 0; x < display_width; ++x) {
                display.set_char(x, 0, '=');
                display.set_char(x, display_height - 1, '=');
            }
            
            // Draw left and right walls with dots
            int left_wall_x = offset_x;
            int right_wall_x = offset_x + game_width;
            for (int y = offset_y; y < offset_y + game_height; ++y) {
                display.set_char(left_wall_x, y, '.');
                display.set_char(right_wall_x, y, '.');
            }
            
            // Draw paddles
            int left_paddle_x = left_wall_x + 1;  // Slightly offset from wall
            int left_paddle_y = offset_y + static_cast<int>((pong.left_pos + pong.width/2) * scale_y);
            int paddle_height = std::max(1, static_cast<int>(pong.paddle_width * scale_y / 2));
            for (int y = left_paddle_y - paddle_height; y <= left_paddle_y + paddle_height; ++y) {
                if (y >= offset_y && y < offset_y + game_height) {
                    display.set_char(left_paddle_x, y, '|');
                }
            }
            
            int right_paddle_x = right_wall_x - 1;  // Slightly offset from wall
            int right_paddle_y = offset_y + static_cast<int>((pong.right_pos + pong.width/2) * scale_y);
            for (int y = right_paddle_y - paddle_height; y <= right_paddle_y + paddle_height; ++y) {
                if (y >= offset_y && y < offset_y + game_height) {
                    display.set_char(right_paddle_x, y, '|');
                }
            }
            
            // Draw ball
            int ball_x = offset_x + static_cast<int>((pong.ball_pos.x + pong.length/2) * scale_x);
            int ball_y = offset_y + static_cast<int>((pong.ball_pos.y + pong.width/2) * scale_y);
            if (ball_x >= offset_x && ball_x < offset_x + game_width && 
                ball_y >= offset_y && ball_y < offset_y + game_height) {
                display.set_char(ball_x, ball_y, 'O');
            }
            
            // Draw score and info
            std::string score_line = "Score: " + std::to_string(pong.left_score) + " - " + std::to_string(pong.right_score);
            for (size_t i = 0; i < score_line.length() && i < display_width; ++i) {
                display.set_char(i, display_height - 2, score_line[i]);
            }
            
            // Render the frame
            display.render();
            
            // Add a small delay - 60 FPS = ~16.67ms
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
            
            pong.tick();
        }
        
        std::cout << "SCORES: " << pong.left_score << ", " << pong.right_score << std::endl << std::endl;

        if (running) {
            std::cout << "Select generation to simulate: ";
            std::cout.flush();
            std::cin >> gen;
        }
    }

    return 0;
} 