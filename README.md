# LiberTEM-rs

This repository contains rust-based plumbing for [LiberTEM](https://github.com/LiberTEM/LiberTEM/)
and [LiberTEM-live](https://github.com/LiberTEM/LiberTEM-live/). The individual packages generally
come with Python bindings using PyO3, and can be built using [maturin](https://github.com/PyO3/maturin/).
The repository is structured as a cargo workspace, and some of the crates are just used internally,
like `bs-sys`.

Minimum supported rust version (MSRV) is 1.74 (November 2023).
Minimum Python version supported is 3.9.

## Development

Please clone using `git clone --recurse-submodules ...` to include vendored
code in submodules. After cloning, remember to enable pre-commit hooks, for example
using `uvx pre-commit install --install-hooks`.

To keep the `THIRDPARTY.yml` files updated, please run
`cargo bundle-licenses --format yaml --output THIRDPARTY.yml`
in the respective crate folders, whenever dependencies or versions change.

### Making a release

- Bump the versions in the Cargo.toml files belonging to the Python packages
  you want to include. Use the same version for all packages.
- Add a small changelog paragraph to their README.md files
- Merge these changes as a PR into the main branch
- Wait for CI to finish for the main branch
- Once this is done, create a new release with a new tag (vX.Y.Z) matching the
  version selected above. Feel free to use the "generate release notes" button
  in the release UI and add a short "human-readable" summary of the release highlights.
- Once the release is published, CI will run another workflow for the new tag,
  and publish the wheels to PyPI and the GitHub release.

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
