from k2opy import Cam, Sync, AcquisitionParams, Writer


def main():
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
        # sync=Sync.WaitForSync,
        sync=Sync.Immediately,
        writer=writer,
        enable_frame_iterator=False,
        shm_path="/tmp/k2shm.socket",
    )

    aq = cam.make_acquisition(aqp)
    aq.arm()

    # only needed if we only write to a file and don't consume the frame iterator:
    aq.wait_until_complete()  
    print("stopping...")
    aq.stop()


if __name__ == "__main__":
    main()
