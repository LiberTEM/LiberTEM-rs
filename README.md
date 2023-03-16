# LiberTEM-rs

This repository contains rust-based plumbing for [LiberTEM](https://github.com/LiberTEM/LiberTEM/)
and [LiberTEM-live](https://github.com/LiberTEM/LiberTEM-live/). The individual packages generally
come with Python bindings using PyO3, and can be built using [maturin](https://github.com/PyO3/maturin/).
The repository is structured as a cargo workspace, and some of the crates are just used internally,
like `bs-sys`.


## Contents

- `bs-sys`: rust bindings to `bitshuffle`.
- `ipc_test`: internal crate for efficient shared memory communication using a shared slab data structure.
- `libertem_asi_tpx3`: A Rust+Python library for receiving sparse array streams from Amsterdam Scientific Instruments CheeTah TPX3 detectors.
- `libertem_dectris`: This is a Python package for efficiently receiving data from DECTRIS detectors with the zeromq interface.
- `playegui`: `egui`-based prototype for efficient on-line visualization of 4D STEM reconstructions
