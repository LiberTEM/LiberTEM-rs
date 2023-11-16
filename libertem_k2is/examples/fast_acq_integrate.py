"""
This example demonstrates fast turnaround of multiple acquisitions, where:

1) Each acquisition is integrating potentially many frames, and
2) There is little down time between acquisitions, and
3) Each acquisition start point is properly synchronized, meaning it doesn't
   include data from any point in time before explicitly starting the
   acquisition.
"""

import time

import click
import numpy as np
from libertem.common.tracing import maybe_setup_tracing

from opentelemetry import trace

from k2opy import (
    Cam, Sync, AcquisitionParams, Writer, CamClient, Mode,
)

tracer = trace.get_tracer("write_and_iterate")


def iterate(aq, cam, cam_client, frame_arr):
    t0 = time.time()
    i = 0
    try:
        while frame := cam.get_next_frame():
            i += 1
            if frame.is_dropped():
                print(f"dropped frame {frame.get_idx()}")
                continue
            slot = cam.get_frame_slot(frame)
            frame_ref = cam_client.get_frame_ref(slot)
            mv = frame_ref.get_memoryview()
            payload = np.frombuffer(mv, dtype=np.uint16).reshape(
                cam.get_frame_shape()
            )
            frame_arr += payload
            cam_client.done(slot)
    finally:
        t1 = time.time()
        print(f"acquisition {aq} done, got {i} frames in {t1-t0:.2f}s...")
        # np.save(f"/cachedata/alex/bar-sum-{aq.get_id()}.npy", frame_arr)


@click.command
@click.option('--mode', default="summit", type=str)
@click.option('--num-parts', default=4, type=int)
@click.option('--frames-per-part', default=10, type=int)
def main(mode, num_parts, frames_per_part):
    maybe_setup_tracing("write_and_iterate")
    with tracer.start_as_current_span("main"):
        if mode.lower() == "summit":
            mode = Mode.Summit
        elif mode.lower() == "is":
            mode = Mode.IS
        shm_socket = "/tmp/k2shm.socket"
        cam = Cam(
            local_addr_top="192.168.10.99",
            local_addr_bottom="192.168.10.99",
            mode=mode,
            enable_frame_iterator=True,
            shm_path=shm_socket,
        )
        cam_client = CamClient(shm_socket)

        frame_arr = np.zeros(cam.get_frame_shape(), dtype=np.float32)

        try:
            for i in range(num_parts):
                frame_arr[:] = 0
                writer = Writer(
                    method="direct",
                    # method="mmap",
                    filename=f"/cachedata/alex/bar-{i}.raw",
                )
                aqp = AcquisitionParams(
                    size=frames_per_part,
                    # sync=Sync.WaitForSync,
                    sync=Sync.Immediately,
                    writer=writer,
                    # writer=None,
                )
                aq = cam.make_acquisition(aqp)
                print(f"acquisition {aq}")
                cam.wait_for_start()
                iterate(aq, cam, cam_client, frame_arr)
        finally:
            # this shuts down the runtime, backgrounds threads an all...
            cam.stop()


if __name__ == "__main__":
    main()
