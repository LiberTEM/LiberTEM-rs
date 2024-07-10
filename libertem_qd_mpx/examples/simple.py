from libertem_qd_mpx import CamClient, QdConnection, QdFrameStack
import numpy as np


if __name__ == "__main__":
    socket_path = "/tmp/qdmpxshm.socket"
    conn = QdConnection(
        data_host="localhost",
        data_port=6342,
        frame_stack_size=16,
        shm_handle_path=socket_path,
    )
    try:
        cam_client = CamClient(handle_path=socket_path)
        conn.start_passive()
        assert conn.wait_for_arm(10) is not None
        buf = np.zeros((32, 256, 256), dtype=np.float32)
        while True:
            stack = conn.get_next_stack(32)
            if stack is None:
                print("done")
                break
            cam_client.decode_range_into_buffer(stack, buf, 0, len(stack))
            cam_client.done(stack)
    finally:
        conn.close()
