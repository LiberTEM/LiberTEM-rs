import time
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient

if __name__ == "__main__":
    # this is emulating a slow data consumer
    # if the number of slots is large enough, this should still work,
    # as the whole acquisition _can_ fit into shared memory
    conn = libertem_dectris.DectrisConnection(
        uri="tcp://localhost:9999",
        handle_path="/tmp/dectris_shm",
        frame_stack_size=32,
        num_slots=1000,
        bytes_per_frame=512*512,
        huge=False,
    )
    ec = DEigerClient('localhost', 8910)

    ec.setDetectorConfig('ntrigger', 1)
    ec.setDetectorConfig('nimages', 1)
    ec.setDetectorConfig('trigger_mode', 'exte')
    ec.setDetectorConfig('ntrigger', 256*256)

    result = ec.sendDetectorCommand('arm')
    sequence_id = result['sequence id']

    conn.start(series=sequence_id)
    cam_client = libertem_dectris.CamClient(conn.get_socket_path())

    while True:
        stack_handle = conn.get_next_stack(max_size=1024)
        if stack_handle is None:
            break
        print(f"{len(stack_handle)}")
        time.sleep(1.1)
        cam_client.done(stack_handle)

    conn.close()
