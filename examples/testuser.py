import time
import click
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient

@click.command()
@click.argument('nimages', type=int)
def main(nimages: int):
    frames = libertem_dectris.FrameIterator()

    ec = DEigerClient('localhost', 8910)

    ec.setDetectorConfig('ntrigger', 1)
    ec.setDetectorConfig('nimages', 1)
    ec.setDetectorConfig('trigger_mode', 'exte')
    ec.setDetectorConfig('ntrigger', nimages)

    result = ec.sendDetectorCommand('arm')
    sequence_id = result['sequence id'] 

    frames.start(series=sequence_id)

    t0 = time.perf_counter()
    for frame in frames:
        pass
    t1 = time.perf_counter()
    print(f"python done in {t1 - t0}; {nimages/(t1 - t0)} fps")

    frames.close()


if __name__ == "__main__":
    main()
