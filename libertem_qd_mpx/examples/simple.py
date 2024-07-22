import time

from libertem_live.detectors.merlin.control import MerlinControl
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
        while True:
            print("starting acquisition...")
            with MerlinControl() as c:
                time.sleep(1)
                c.cmd('STARTACQUISITION')
                c.cmd('SOFTTRIGGER')
            header = conn.wait_for_arm(10)
            print(header)
            assert header is not None
            buf = np.zeros((32, 512, 512), dtype=np.float32)
            cbed = np.zeros((512, 512), dtype=np.float32)
            t0 = time.perf_counter()
            while True:
                stack = conn.get_next_stack(32)
                if stack is None:
                    print("done")
                    break
                view = buf[:len(stack)]
                cam_client.decode_range_into_buffer(stack, view, 0, len(stack))
                cam_client.done(stack)
                cbed += view.sum(axis=0)
            t1 = time.perf_counter()
            print(t1 - t0)
    finally:
        conn.close()
