import numpy as np
import tqdm
import libertem_asi_mpx3

conn = libertem_asi_mpx3.ServalConnection(
    data_uri="localhost:8283",
    api_uri="http://localhost:8080",
    handle_path="/tmp/asi_mpx3_shm",
    frame_stack_size=16,
    num_slots=2000,
    bytes_per_frame=512*512*2,
    huge=False,
)

conn.start_passive()

cam_client = None

try:
    while True:
        config = None
        while config is None:
            print("connecting...")
            config = conn.wait_for_arm(10.0)

        assert config is not None
        print(config)

        # any other process can use a `CamClient` to use data
        # stored in the SHM:
        cam_client = libertem_asi_mpx3.CamClient(conn.get_socket_path())

        tq = tqdm.tqdm(total=config.get_n_triggers() * 512 * 512 * 2, unit='B', unit_scale=True, unit_divisor=1024)

        while True:
            # get at most `max_size` frames as a stack
            # (might get less at the end of the acquisition)
            stack_handle = conn.get_next_stack(max_size=32)

            # if the receiver is idle, stack_handle will be None here:
            if stack_handle is None:
                break

            # the expected shape and data type:
            frame_shape = tuple(stack_handle.get_shape())

            frames = cam_client.get_frames(stack_handle)

            tq.update(len(frames) * 512 * 512 * 2)

            del frames  # let's hope no-one else keeps a reference, as it will be invalid after `done` is called

            # free up the shared memory slot for this frame stack:
            cam_client.done(stack_handle)

        tq.close()
finally:
    conn.close()  # clean up background thread etc.
    if cam_client is not None:
        cam_client.close()
