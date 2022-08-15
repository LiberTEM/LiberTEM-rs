import time
import perf_utils
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient

if __name__ == "__main__":
    ec = DEigerClient("localhost", 8910)
    frames = libertem_dectris.FrameChunkedIterator()

    frames.start()

    nimages = 256 * 256
    ec.setDetectorConfig('ntrigger', 1)
    ec.setDetectorConfig('nimages', 1)
    ec.setDetectorConfig('trigger_mode', 'exte')
    ec.setDetectorConfig('ntrigger', nimages)
    result = ec.sendDetectorCommand('arm')
    sequence_id = result['sequence id']

    with perf_utils.perf('frame_iterator', output_dir='profiles') as perf_data:
        t0 = time.perf_counter()
        while frames.is_running():
            stack = frames.get_next_stack(max_size=32)
            s = stack.serialize()
            new_stack = libertem_dectris.FrameStack.deserialize(s)
            # for i in range(len(new_stack)):
            #     frame = new_stack[i]
        t1 = time.perf_counter()
        print(t1-t0)

    frames.close()
