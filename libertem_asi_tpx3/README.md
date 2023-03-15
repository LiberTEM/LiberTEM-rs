# LiberTEM-asi-tpx3

A Rust+Python library for receiving sparse array streams from Amsterdam
Scientific Instruments CheeTah TPX3 detectors.

# Development

Needs to have a Python environment active, and a recent rust toolchain installed.

- `pip install maturin[patchelf] numpy`
- `maturin develop -r`
- `python examples/simple.py`
