#!/bin/bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example consumer -- --socket-path=/tmp/some-sock "$@"
