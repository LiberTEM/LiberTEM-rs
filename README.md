# LiberTEM-dectris-rs

This is a Python package for efficiently receiving data from DECTRIS detectors
with [the zeromq interface](https://media.dectris.com/210607-DECTRIS-SIMPLON-API-Manual_EIGER2-chip-based_detectros.pdf).
The low-level, high-frequency operations are performed in a background thread
implemented in rust, and multiple frames are batched together for further
processing in Python.

Decoding of compressed frames is not (yet) handled in this package, but may be
added later.

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

frames = libertem_dectris.FrameChunkedIterator()
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
            # do something with the frame; compression
            # is not handled in this module (yet)
            image_data_bytes = frame.get_image_data()
            shape = frame.get_shape()
            encoding = frame.get_encoding()
            frame_id = frame.get_frame_id()
        if len(stack) == 0:
            break
finally:
    frames.close()  # clean up background thread etc.
```
