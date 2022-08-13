import time
import rusted_dectris

if __name__ == "__main__":
    frames = rusted_dectris.FrameIterator()

    frames.start(series=14)

    for frame in frames:
        time.sleep(1.1)

    frames.close()

