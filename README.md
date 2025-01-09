# LiberTEM-rs

This repository contains rust-based plumbing for [LiberTEM](https://github.com/LiberTEM/LiberTEM/)
and [LiberTEM-live](https://github.com/LiberTEM/LiberTEM-live/). The individual packages generally
come with Python bindings using PyO3, and can be built using [maturin](https://github.com/PyO3/maturin/).
The repository is structured as a cargo workspace, and some of the crates are just used internally,
like `bs-sys`.

Minimum supported rust version (MSRV) is 1.72 (August 2023)

## Development

Please clone using `git clone --recurse-submodules ...` to include vendored
code in submodules. After cloning, remember to enable pre-commit hooks using
`pre-commit install --install-hooks`.

## Contents

- `bs-sys`: rust bindings to `bitshuffle`.
- `ipc_test`: internal crate for efficient shared memory communication using a shared slab data structure.
- `common`: generic traits, types and macros for supporting detectors
- `libertem_asi_tpx3`: A Rust+Python library for receiving sparse array streams from Amsterdam Scientific Instruments CheeTah TPX3 detectors.
- `libertem_asi_mpx3`: A Rust+Python library for receiving data from Amsterdam Scientific Instruments frame-based detectors (experimental).
- `serval-client`: A rust crate for speaking to the ASI Serval API
- `libertem_dectris`: This is a Python package for efficiently receiving data from DECTRIS detectors with the zeromq interface.
- `libertem_qd_mpx`: A Rust+Python library for receiving data from Quantum Detectors MerlinEM detectors.
- `playegui`: `egui`-based prototype for efficient on-line visualization of 4D STEM reconstructions


## License

All crates are made available under the MIT license, if not specified otherwise.
