# Makefile for Pong Neuroevolution Project

# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
DEBUG_FLAGS = -g -DDEBUG
RELEASE_FLAGS = -O3 -DNDEBUG

# Directories
SRC_DIR = src
APPS_DIR = apps
BUILD_DIR = build
BIN_DIR = bin
CORE_DIR = $(SRC_DIR)/core
AI_DIR = $(SRC_DIR)/ai
PLAYERS_DIR = $(AI_DIR)/players

# Object files
CORE_OBJS = $(BUILD_DIR)/geometry.o $(BUILD_DIR)/pong.o
AI_OBJS = $(BUILD_DIR)/neural_net.o $(BUILD_DIR)/evolution.o
PLAYER_OBJS = $(BUILD_DIR)/neural_player.o $(BUILD_DIR)/simple_player.o
ALL_OBJS = $(CORE_OBJS) $(AI_OBJS) $(PLAYER_OBJS)

# Executable names
TRAIN_EXE = $(BIN_DIR)/pong_evolution
REPLAY_EXE = $(BIN_DIR)/pong_replay
TEST_EXE = $(BIN_DIR)/test_display

# Default target
all: $(TRAIN_EXE) $(REPLAY_EXE) $(TEST_EXE)

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Core object files
$(BUILD_DIR)/geometry.o: $(CORE_DIR)/geometry.cpp $(CORE_DIR)/geometry.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/pong.o: $(CORE_DIR)/pong.cpp $(CORE_DIR)/pong.hpp $(CORE_DIR)/geometry.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# AI object files
$(BUILD_DIR)/neural_net.o: $(AI_DIR)/neural_net.cpp $(AI_DIR)/neural_net.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/evolution.o: $(AI_DIR)/evolution.cpp $(AI_DIR)/evolution.hpp $(AI_DIR)/neural_net.hpp $(CORE_DIR)/pong.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Player object files
$(BUILD_DIR)/neural_player.o: $(PLAYERS_DIR)/neural_player.cpp $(PLAYERS_DIR)/neural_player.hpp $(AI_DIR)/neural_net.hpp $(CORE_DIR)/pong.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/simple_player.o: $(PLAYERS_DIR)/simple_player.cpp $(PLAYERS_DIR)/simple_player.hpp $(CORE_DIR)/pong.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Executables
$(TRAIN_EXE): $(APPS_DIR)/train_evolution.cpp $(ALL_OBJS) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(ALL_OBJS) -o $@

$(REPLAY_EXE): $(APPS_DIR)/replay_player.cpp $(CORE_OBJS) $(BUILD_DIR)/neural_player.o $(BUILD_DIR)/neural_net.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(CORE_OBJS) $(BUILD_DIR)/neural_player.o $(BUILD_DIR)/neural_net.o -o $@

$(TEST_EXE): $(APPS_DIR)/test_display.cpp $(CORE_OBJS) $(BUILD_DIR)/simple_player.o | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $< $(CORE_OBJS) $(BUILD_DIR)/simple_player.o -o $@

# Debug builds
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: all

# Release builds
release: CXXFLAGS += $(RELEASE_FLAGS)
release: all

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) fittest.log

# Test build (just compile, don't run)
test-build: all
	@echo "All targets built successfully!"

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Build all executables (default)"
	@echo "  debug        - Build with debug flags"
	@echo "  release      - Build with release optimization"
	@echo "  clean        - Remove all build artifacts"
	@echo "  test-build   - Test that everything compiles"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Executables:"
	@echo "  $(TRAIN_EXE)   - Main neuroevolution training program"
	@echo "  $(REPLAY_EXE)     - Replay/visualize evolved players"
	@echo "  $(TEST_EXE)       - Demo with improved terminal display"

.PHONY: all debug release clean test-build help 