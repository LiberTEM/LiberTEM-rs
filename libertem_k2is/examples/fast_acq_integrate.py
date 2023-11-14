"""
This example demonstrates fast turnaround of multiple acquisitions, where:

1) Each acquisition is integrating potentially many frames, and
2) There is little down time between acquisitions, and 
3) Each acquisition start point is properly synchronized, meaning it doesn't
   include data from any point in time before explicitly starting the
   acquisition.
"""

import time

import numpy as np
from libertem.common.buffers import zeros_aligned
from libertem.common.tracing import maybe_setup_tracing

from opentelemetry import trace

from k2opy import (
    Cam, Sync, AcquisitionParams, Writer, CamClient, Mode,
)

tracer = trace.get_tracer("write_and_iterate")


def iterate(aq, cam_client):
    frame_arr = zeros_aligned(aq.get_frame_shape(), dtype=np.uint16)
    t0 = time.time()
    i = 0
    try:
        while frame := aq.get_next_frame():
            i += 1
            if frame.is_dropped():
                print(f"dropped frame {frame.get_idx()}")
                continue
            slot = aq.get_frame_slot(frame)
            frame_ref = cam_client.get_frame_ref(slot)
            mv = frame_ref.get_memoryview()
            payload = np.frombuffer(mv, dtype=np.uint16).reshape(
                aq.get_frame_shape()
            )

            payload.sum()
            # print(payload.sum())

            cam_client.done(slot)
            # print(arr.sum(), arr.shape, arr.dtype)
    finally:
        t1 = time.time()
        print(f"stopping, got {i} frames in {t1-t0:.2f}s...")
        aq.stop()


def main():
    maybe_setup_tracing("write_and_iterate")
    with tracer.start_as_current_span("main"):
        cam = Cam(
            local_addr_top="192.168.10.99",
            local_addr_bottom="192.168.10.99",
        )

        # writer = Writer(
        #     method="direct",
        #     # method="mmap",
        #     filename="/cachedata/alex/bar.raw",
        # )
        shm_socket = "/tmp/k2shm.socket"
        aqp = AcquisitionParams(
            size=18,
            # sync=Sync.WaitForSync,
            sync=Sync.Immediately,
            # writer=writer,
            writer=None,
            enable_frame_iterator=True,
            shm_path=shm_socket,
            mode=Mode.Summit,
        )

        aq = cam.make_acquisition(aqp)
        aq.arm()

        cam_client = CamClient(shm_socket)

        aq.wait_for_start()
        iterate(aq, cam_client)

        aq.wait_for_start()
        iterate(aq, cam_client)


if __name__ == "__main__":
    main()
