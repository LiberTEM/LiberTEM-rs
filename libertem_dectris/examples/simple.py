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
