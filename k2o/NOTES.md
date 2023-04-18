# SIMD

- bounds check
    - use of `chunks_exact` elides bounds checks!
- need to explicitly enable the right CPU target (for example: `RUSTFLAGS="-C target-cpu=native"`
    - see also `.cargo/config.toml`
- for unrolling, might need to nest the loops

# Writing to disk

- first attempt, just to get something working
    - mpsc queue, have a single thread writing to disk
    - => slow, because this single thread limits the throughput
- second attempt, one writer thread per receiver thread
    - still slow (or even slower!) - most of the time is spent in the page
      fault handler of the kernel
- experiments with pre-allocating the disk space
    - using fallocate, also re-using the npy file from a previous run
    - => no change in performance, still spending 88% of our time in the page fault handler
- assembly of full frames is required to get a good access pattern!

# TODO
- [ ] fix busy waiting (one of the `try_recv` loops, I guess)
- [ ] fix subframe shtuff
