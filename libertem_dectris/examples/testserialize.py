import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient

if __name__ == "__main__":
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

    conn.start(sequence_id)

    cam_client = libertem_dectris.CamClient(conn.get_socket_path())
    stack_handle = conn.get_next_stack(max_size=32)
    serialized = stack_handle.serialize()
    loaded = libertem_dectris.FrameStackHandle.deserialize(serialized)

    print(serialized)
    print(len(loaded))
    print(loaded.get_pixel_type())
    print(loaded.get_encoding())
    print(loaded.get_shape())

    cam_client.done(stack_handle)
    conn.close()