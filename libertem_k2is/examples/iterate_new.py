import time

import numpy as np
from libertem.common.buffers import zeros_aligned
from libertem.common.tracing import maybe_setup_tracing

from opentelemetry import trace

from libertem_k2is import (
    K2Connection, K2AcquisitionConfig, K2CamClient, K2Mode,
)

tracer = trace.get_tracer("write_and_iterate")


def main():
    maybe_setup_tracing("iterate_new")
    with tracer.start_as_current_span("main"):
        shm_socket = "/tmp/k2shm.socket"
        conn = K2Connection(
            local_addr_top="192.168.10.98",
            local_addr_bottom="192.168.10.99",
            shm_handle_path=shm_socket,
            frame_stack_size=1,
            mode=K2Mode.Summit,
        )

        frame_shape = conn.get_frame_shape()
        conn.start_passive()
        cam_client = K2CamClient(shm_socket)
        frame_arr = zeros_aligned((1,) + frame_shape, dtype=np.uint16)

        conn.wait_for_arm()

        t0 = time.time()
        i = 0
        try:
            while stack := conn.get_next_stack(1):
                print(f"{i}: {stack}")
                i += 1
                try:
                    cam_client.decode_range_into_buffer(
                        stack,
                        frame_arr,
                        0,
                        1,
                    )
                finally:
                    cam_client.frame_stack_done(stack)
        except Exception as e:
            print(f"got an exception: {e}")
            raise
        finally:
            t1 = time.time()
            print(f"stopping, got {i} frames in {t1-t0:.2f}s...")
            conn.close()


if __name__ == "__main__":
    main()
