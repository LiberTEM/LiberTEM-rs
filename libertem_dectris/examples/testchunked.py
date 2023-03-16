import time
import click
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient


@click.command()
@click.argument('nimages', type=int)
def main(nimages: int):
    ec = DEigerClient('localhost', 8910)

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

    try:
        t0 = time.perf_counter()
        total = 0
        while True:
            stack_handle = conn.get_next_stack(max_size=32)
            if stack_handle is None:
                break
            stack_handle.serialize()
            total += len(stack_handle)
            cam_client.done(stack_handle)
    finally:
        t1 = time.perf_counter()
        print(f"got {total} frames in {t1 - t0:.3f}s ({total/(t1-t0):.3f} fps); done, closing")
        conn.close()


if __name__ == "__main__":
    main()
