# Running the examples

The Linux example requires hugepages:

```
$ sudo sysctl -w vm.nr_hugepages=102400
```

```bash
# consumer
RUSTFLAGS="-C target-cpu=native" cargo run --example consumer -- --socket-path=/tmp/some-sock

# producer
RUSTFLAGS="-C target-cpu=native" cargo run --example producer -- --socket-path=/tmp/some-sock
```
