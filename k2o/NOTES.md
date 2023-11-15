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

# Notes on reworking the life cycles

To make sure we can quickly start acquisitions after another,
we need to make sure of some boundary conditions:

- The SHM area needs to be kept alive over multiple acquisitions, as well as
  the "recycling" system for blocks, such that we don't have malloc/pagefault
  caused jitter
- The sockets need to be kept open, otherwise the multicast traffic may only be
  received after an initial delay (IIRC)
- That also means that switching between more fundamental settings may take longer
    - changing network settings
    - changing camera mode (IS/Summit)
    - SHM socket path
    - enabling / disabling the frame iterator
- What changes must be fast:
    - file writer destination (filename) -> we may need to disable pre-allocation of files!
    - number of frames per acquisition
    - probably: camera sync mode (immediately vs. wait for sync flag)

## Action items
- [ ] Refactor...
    - [x] The `AcquisitionRuntime` should be the object that lives over multiple acquisitions
    - [x] On the Python side, the `Cam` already needs to start the `AcquisitionRuntime`
    - [ ] For error handling, we should have the option to completely tear down
      and re-create the `AcquisitionRuntime`, perform this step automatically such that
      we don't have "restart the {script,notebook,app,server}" situations as often as currently
    - [x] `WriterBuilder` must be set per acquisition, not when starting the runtime
- [x] Figure out where the acquisition ID should be generated
    - probably in the `AcquisitionRuntime`, as that is the long-living object
      which manages the background thread(s) etc.

# TODO
- [ ] fix busy waiting (one of the `try_recv` loops, I guess)
- [ ] fix subframe shtuff
