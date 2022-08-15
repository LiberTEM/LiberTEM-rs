import time
import click
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient


@click.command()
@click.argument('nimages', type=int)
def main(nimages: int):
    frames = libertem_dectris.FrameChunkedIterator()

    ec = DEigerClient('localhost', 8910)

    ec.setDetectorConfig('ntrigger', 1)
    ec.setDetectorConfig('nimages', 1)
    ec.setDetectorConfig('trigger_mode', 'exte')
    ec.setDetectorConfig('ntrigger', nimages)

    result = ec.sendDetectorCommand('arm')
    sequence_id = result['sequence id'] 

    frames.start(series=sequence_id)

    try:
        t0 = time.perf_counter()
        total = 0
        while True:
            stack = frames.get_next_stack(max_size=32)
            stack.serialize()
            total += len(stack)
            if len(stack) == 0:
                break
    finally:
        t1 = time.perf_counter()
        print(f"got {total} frames in {t1 - t0:.3f}s ({total/(t1-t0):.3f} fps); done, closing")
        frames.close()


if __name__ == "__main__":
    main()
