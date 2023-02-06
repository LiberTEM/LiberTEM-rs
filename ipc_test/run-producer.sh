#!/bin/bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example producer -- --socket-path=/tmp/some-sock "$@"
