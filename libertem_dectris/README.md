# LiberTEM-dectris-rs

[![LiberTEM-dectris-rs on GitHub](https://img.shields.io/badge/GitHub-MIT-informational)](https://github.com/LiberTEM/LiberTEM-rs)

This is a Python package for efficiently receiving data from DECTRIS detectors
with [the zeromq interface](https://media.dectris.com/210607-DECTRIS-SIMPLON-API-Manual_EIGER2-chip-based_detectros.pdf).
The low-level, high-frequency operations are performed in a background thread
implemented in rust, and multiple frames are batched together for further
processing in Python.

It is built for [LiberTEM-live](https://github.com/libertem/libertem-live), but can
also be used stand-alone.

## Usage

```python
import numpy as np
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient

# trigger acquisition via the REST API, needs `libertem-live`
nimages = 256 * 256
ec = DEigerClient('localhost', 8910)  # hostname and port of the DCU REST API
ec.setDetectorConfig('ntrigger', 1)
ec.setDetectorConfig('nimages', 1)
ec.setDetectorConfig('trigger_mode', 'exte')
ec.setDetectorConfig('ntrigger', nimages)
result = ec.sendDetectorCommand('arm')
sequence_id = result['sequence id'] 

conn = libertem_dectris.DectrisConnection(
    uri="tcp://localhost:9999",
    handle_path="/tmp/dectris_shm",
    frame_stack_size=32,
    num_slots=2000,
    bytes_per_frame=512*512,
    huge=False,
)

# as we have armed the detector above, we know the sequence number
# that we should expect:
# (in other cases, can also call
# `conn.start_passive` and `conn.wait_for_arm` to passively wait for
# the detector to be armed)
conn.start(sequence_id)

# any other process can use a `CamClient` to use data
# stored in the SHM:
cam_client = libertem_dectris.CamClient(conn.get_socket_path())

try:
    while True:
        # get at most `max_size` frames as a stack
        # (might get less at the end of the acquisition)
        stack_handle = conn.get_next_stack(max_size=32)

        # if the receiver is idle, stack_handle will be None here:
        if stack_handle is None:
            break

        # the expected shape and data type:
        frame_shape = tuple(reversed(stack_handle.get_shape()))
        dtype = np.dtype(stack_handle.get_pixel_type()).newbyteorder(
            stack_handle.get_endianess()
        )

        # pre-allocate some memory for the pixel data:
        # (would be pulled out of the loop in real code)
        buf = np.zeros((len(stack_handle),) + frame_shape, dtype=dtype)

        # decompress into the pre-allocated buffer
        cam_client.decompress_frame_stack(stack_handle, out=buf)

        # free up the shared memory slot for this frame stack:
        cam_client.done(stack_handle)

        # we can still use the decompressed data:
        buf.sum()
finally:
    conn.close()  # clean up background thread etc.
    cam_client.close()

```

## Changelog

### v0.2.12

- Refactored and unified interface - allows for more code sharing between crates
- Some robustness changes around adding timeout parameters for many top-level operations

### v0.2.11

- Re-release for CI changes

### v0.2.10

- Add functionality to generate mock data for sending via the simulator
- Add missing numpy dependency
- Make debug output a bit more reliable

### v0.2.9

- Add more debug and trace output

### v0.2.7

- Log more details in `DectrisConnection.log_shm_stats` and change log level to
  `INFO`
- Increase timeout for sending headers and frames

### v0.2.4 - v0.2.6

- Updated examples and CI configuration

### v0.2.3

- Add `env_logger`: set environment variable `LIBERTEM_DECTRIS_LOG_LEVEL` to e.g. `'INFO'` to enable logging
- Improved error handling: raise an exception instead of panicing on serialization errors
- Ignore messages with mismatching series ID
- Add explicit checks for the correct `header_detail` levels
- Move code into monorepo

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
