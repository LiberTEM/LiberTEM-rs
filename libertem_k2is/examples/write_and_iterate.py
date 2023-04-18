import numpy as np
from libertem.common.buffers import zeros_aligned
from libertem.common.tracing import maybe_setup_tracing

from opentelemetry import trace

from k2opy import Cam, Sync, AcquisitionParams, Writer

tracer = trace.get_tracer("write_and_iterate")


def main():
    maybe_setup_tracing("write_and_iterate")
    with tracer.start_as_current_span("main"):
        cam = Cam(
            local_addr_top="192.168.10.99",
            local_addr_bottom="192.168.10.99",
        )

        writer = Writer(
            method="direct",
            # method="mmap",
            filename="/cachedata/alex/bar.raw",
        )
        aqp = AcquisitionParams(
            size=1800,
            sync=Sync.WaitForSync,
            # sync=Sync.Immediately,
            # writer=writer,
            writer=None,
            enable_frame_iterator=True,
        )

        aq = cam.make_acquisition(aqp)
        aq.arm()

        frame_arr = zeros_aligned((2048, 1860), dtype=np.uint16)
        frame_indexes = set()

        aq.wait_for_start()

        i = 0
        try:
            while frame := aq.get_next_frame():
                i += 1
                # allocating here is a no-no:
                # arr = frame.get_array()
                # print(frame.do_stuff())
                frame_indexes.add(frame.get_idx())
                if frame.is_dropped():
                    print(f"dropped frame {frame.get_idx()}")
                    continue
                frame.get_array_into(out=frame_arr)
                # frame_arr.sum()
                # print(frame_arr.sum())
                aq.frame_done(frame)
                # print(arr.sum(), arr.shape, arr.dtype)
        finally:
            print(f"stopping, got {i} frames...")
            aq.stop()
        missing = set(range(1800)) - frame_indexes
        print(missing, len(missing))


if __name__ == "__main__":
    main()
