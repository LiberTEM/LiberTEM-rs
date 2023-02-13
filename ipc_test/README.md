# Running the examples

```bash
# consumer
RUSTFLAGS="-C target-cpu=native" cargo run --example consumer -- --socket-path=/tmp/some-sock

# producer
RUSTFLAGS="-C target-cpu=native" cargo run --example producer -- --socket-path=/tmp/some-sock
```
