#!/bin/bash
rm -rf dist/*
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/pong_ai.wasm dist/
cp assets/* dist/
# wasm-pack build --release --target web --out-dir dist/
basic-http-server dist/
