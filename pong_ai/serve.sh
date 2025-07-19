#!/bin/bash
set -e

TARGET=wasm32-unknown-unknown
OUT_DIR=out

rm -rf $OUT_DIR
mkdir -p $OUT_DIR

RUSTFLAGS='--cfg=getrandom_backend="wasm_js"' cargo build --release --target wasm32-unknown-unknown

wasm-bindgen --target web --no-typescript --out-dir $OUT_DIR target/$TARGET/release/pong_ai.wasm

cp -r assets/* $OUT_DIR/

basic-http-server $OUT_DIR
