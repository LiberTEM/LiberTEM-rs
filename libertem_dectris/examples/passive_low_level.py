import tqdm
import numpy as np
import click
import libertem_dectris


@click.command()
@click.argument('width', type=int, default=256)
@click.argument('height', type=int, default=256)
def main(width: int, height: int):
    conn = libertem_dectris.DectrisConnection(
        uri="tcp://localhost:9999",
        handle_path="/tmp/dectris_shm",
        frame_stack_size=32,
        num_slots=2000,
        bytes_per_frame=32*512,
        huge=False,
    )

    conn.start_passive()

    print("now arm and trigger the detector")
    while (res := conn.wait_for_arm(timeout=1.0)) is None:
        print("waiting...")

    config, series = res
    print(f"series {series}; config: {config}")

    cam_client = libertem_dectris.CamClient(conn.get_socket_path())

    tq = tqdm.tqdm(total=config.get_num_frames())

    try:
        while True:
            # get at most `max_size` frames as a stack
            # (might get less at the end of the acquisition)
            stack_handle = conn.get_next_stack(max_size=1024)

            # if the receiver is idle, stack_handle will be None here:
            if stack_handle is None:
                break

            tq.update(len(stack_handle))
            continue

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
    print("done")


if __name__ == "__main__":
    main()

