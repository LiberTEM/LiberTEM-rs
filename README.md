# LiberTEM-dectris-rs

[![LiberTEM-dectris-rs on GitHub](https://img.shields.io/badge/GitHub-MIT-informational)](https://github.com/LiberTEM/LiberTEM-dectris-rs)

This is a Python package for efficiently receiving data from DECTRIS detectors
with [the zeromq interface](https://media.dectris.com/210607-DECTRIS-SIMPLON-API-Manual_EIGER2-chip-based_detectros.pdf).
The low-level, high-frequency operations are performed in a background thread
implemented in rust, and multiple frames are batched together for further
processing in Python.

## Usage

```python
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient

# trigger acquisition via the REST API, needs `libertem-live`
nimages = 512 * 512
ec = DEigerClient('localhost', 8910)  # hostname and port of the DCU REST API
ec.setDetectorConfig('ntrigger', 1)
ec.setDetectorConfig('nimages', 1)
ec.setDetectorConfig('trigger_mode', 'exte')
ec.setDetectorConfig('ntrigger', nimages)
result = ec.sendDetectorCommand('arm')
sequence_id = result['sequence id'] 

frames = libertem_dectris.FrameChunkedIterator(uri="tcp://localhost:9999")
# start to receive data for the given series
# (can be called multiple times on the same `FrameChunkedIterator` instance)
frames.start(series=sequence_id)

try:
    while True:
        # get at most `max_size` frames as a stack
        # (might get less at the end of the acquisition)
        stack = frames.get_next_stack(max_size=32)
        for i in range(len(stack)):
            frame = stack[i]
            image_data_bytes = frame.get_image_data()  # this is the raw data
            shape = frame.get_shape()
            pixel_type = frame.get_pixel_type()  # uint8 etc.
            endianess = frame.get_endianess()  # in numpy notation; ">" or "<"
            dtype = endianess + pixel_type
            encoding = frame.get_encoding()
            frame_id = frame.get_frame_id()
            decompressed = np.zeros(shape, dtype=dtype)
            frame.decompress_into(decompressed)
            # `decompressed` now contains the array data for this frame:
            decompressed.sum()
        if len(stack) == 0:
            break
finally:
    frames.close()  # clean up background thread etc.
```

## Changelog

### v0.2.2

- Vendor `bitshuffle` and add `Frame.decompress_into` method, PR [#10](https://github.com/LiberTEM/LiberTEM-dectris-rs/pull/10)

### v0.2.1

- Catch frame ID mismatch, PR [#9](https://github.com/LiberTEM/LiberTEM-dectris-rs/pull/9)

### v0.2.0

- Added `libertem_dectris.headers` submodule that exports header classes
- Added ways to create `libertem_dectris.Frame` and `libertem_dectris.FrameStack`
  objects from Python, mostly useful for testing
- Added binding to random port for the simulator
- Properly parametrize with zmq endpoint URI
- Fix many clippy complaints

### v0.1.0

Initial release!

## Development

This package is using [pyo3](https://pyo3.rs/) with
[maturin](https://maturin.rs/) to create the Python bindings.  First, make sure
`maturin` is installed in your Python environment:

```bash
(venv) $ pip install maturin
```

Then, after each change to the rust code, run `maturin develop -r` to build and
install a new version of the wheel.

As we vendor `bitshuffle`, make sure to clone with `git clone --recursive ...`, or manually
[take care of initializing and updating submodules](https://github.blog/2016-02-01-working-with-submodules/).

## Release

- update changelog above
- bump version in Cargo.toml if not already bumped, and push
- create a release from the GitHub UI, creating a new tag vX.Y.Z
- done!
