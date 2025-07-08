#!/bin/bash

# Build script for Pong Neuroevolution Project
# This script provides a simple interface to the Makefile

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_usage() {
    echo "Usage: $0 [target]"
    echo ""
    echo "Available targets:"
    echo "  all (default)  - Build all executables"
    echo "  debug          - Build with debug flags"
    echo "  release        - Build with release optimization"
    echo "  clean          - Remove all build artifacts"
    echo "  test           - Test that everything compiles"
    echo "  help           - Show this help message"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if make is available
if ! command -v make >/dev/null 2>&1; then
    print_error "make is not installed or not in PATH"
    exit 1
fi

# Check if g++ is available
if ! command -v g++ >/dev/null 2>&1; then
    print_error "g++ is not installed or not in PATH"
    exit 1
fi

# Default target
TARGET=${1:-all}

case $TARGET in
    all|debug|release|clean|test-build)
        print_info "Building target: $TARGET"
        make $TARGET
        ;;
    test)
        print_info "Running test build"
        make test-build
        ;;
    help)
        print_usage
        ;;
    *)
        print_error "Unknown target: $TARGET"
        print_usage
        exit 1
        ;;
esac

if [ $? -eq 0 ] && [ "$TARGET" != "clean" ] && [ "$TARGET" != "help" ]; then
    print_info "Build completed successfully!"
    echo ""
    echo "Generated executables:"
    if [ -f "pong_evolution" ]; then
        echo "  - pong_evolution  (main training program)"
    fi
    if [ -f "pong_replay" ]; then
        echo "  - pong_replay     (visualization tool)"
    fi
    if [ -f "pong_demo" ]; then
        echo "  - pong_demo       (simple demo)"
    fi
fi 