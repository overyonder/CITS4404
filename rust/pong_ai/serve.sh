#!/bin/bash
rm -rf docs/*
cargo build --release --target wasm32-unknown-unknown
cp target/wasm32-unknown-unknown/release/pong_ai.wasm docs/
cp assets/* docs/
basic-http-server docs/
