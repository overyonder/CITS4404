#include "../src/core/pong.hpp"
#include "../src/ai/players/simple_player.hpp"
#include <iostream>
#include <thread>
#include <chrono>
#include <string>
#include <algorithm>
#include <csignal>

// Global flag for clean shutdown
volatile bool running = true;

// Signal handler for clean shutdown
void signal_handler(int signal) {
    (void)signal;  // Suppress unused parameter warning
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
    
    void render() {
        clear_screen();
        for (const auto& line : buffer) {
            std::cout << line << std::endl;
        }
    }
};

int main() {
    // Set up signal handler for clean shutdown
    signal(SIGINT, signal_handler);
    
    std::cout << "Testing improved terminal display..." << std::endl;
    std::cout << "Press Enter to start demo..." << std::endl;
    std::cin.get();
    
    // Create simple AI players
    SimplePlayer left_player;
    SimplePlayer right_player;
    
    // Create game
    PongGame pong(left_player, right_player);
    pong.max_score = 3;
    
    // Create display buffer
    const int display_width = 80;
    const int display_height = 25;
    TerminalDisplay display(display_width, display_height);
    
    std::cout << "Demo starting in 3 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(3));
    
    // Run demo
    while (running && std::max(pong.left_score, pong.right_score) < pong.max_score) {
        display.clear();
        
        // Calculate proper scaling to maintain aspect ratio
        // Game aspect ratio: length/width = 400/300 = 1.33 (wider than tall)
        // Display aspect ratio: width/height = 80/25 = 3.2
        // We need to fit the game within the display while maintaining proportions
        
        double game_aspect = static_cast<double>(pong.length) / pong.width;  // 1.33
        
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
    
    std::cout << "Demo complete! Final score: " << pong.left_score << " - " << pong.right_score << std::endl;
    return 0;
} 