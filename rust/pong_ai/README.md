# Pong AI

A neural network-powered Pong game built with Rust and WebAssembly.

## Live Demo

The game is deployed on GitHub Pages: [https://yourusername.github.io/pong_ai/](https://yourusername.github.io/pong_ai/)

## Setup for GitHub Pages

This project is configured to deploy to GitHub Pages from the `../docs/` directory. The deployment happens automatically when you push to the `main` branch.

### Prerequisites

1. Make sure your repository is public (or you have GitHub Pro for private repos)
2. Enable GitHub Pages in your repository settings

### Steps to Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**
5. The workflow will automatically deploy from the `../docs/` directory

### Manual Deployment

If you need to deploy manually:

1. Go to **Actions** tab in your repository
2. Select the **Deploy to GitHub Pages** workflow
3. Click **Run workflow** and select the branch

## Local Development

To run the project locally:

```bash
# Install wasm-pack if you haven't already
cargo install wasm-pack

# Build for web
wasm-pack build --target web

# Serve locally (if you have the serve.sh script)
./serve.sh
```

## Project Structure

- `src/` - Rust source code
- `../docs/` - Built web assets (deployed to GitHub Pages)
- `assets/` - Static assets
- `.github/workflows/` - GitHub Actions deployment configuration

## Controls

- **W/S** - Left paddle (in human vs human mode)
- **Up/Down arrows** - Right paddle (in human vs human mode)
- **N** - Toggle debug info
- **P** - Cycle through control modes (AI vs AI, AI vs Human, Human vs Human)
- **Q** - Quit
