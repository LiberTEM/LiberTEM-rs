import time
import libertem_dectris

if __name__ == "__main__":
    frames = libertem_dectris.FrameIterator()

    frames.start(series=14)

    for frame in frames:
        time.sleep(1.1)

    frames.close()

