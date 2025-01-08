from k2opy import K2ISCam, Sync, Flags

cam = K2ISCam()

aq = cam.make_acquisition()
aq.set_sync(Sync.WaitForSync)  # or Sync.Immediately
# aq.set_binning(...) - not implemented fully
aq.set_num_frames(256*256)
# aq.set_num_frames(Flags.Continuous) - alternatively
aq.enable_file_writing(method="direct", dest_path="")
aq.enable_frame_iterator()

it = aq.get_frame_iterator(ordered=True)

def get_frames():
    try:
        shape = aq.get_shape()  # different frame sizes when binning/cropping
        aq.arm()
        for i in range(256*256):
            frame = aq.get_next_frame()
            arr = frame.get_array()
            yield arr
            aq.frame_done(frame)
    finally:
        aq.stop()