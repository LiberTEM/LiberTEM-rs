from libertem_asi_tpx3 import ASITpx3Connection, CamClient

if __name__ == "__main__":
    conn = ASITpx3Connection(
        uri="localhost:8283",
        frame_stack_size=2000,
        bytes_per_frame=1024*1024,
        huge=False,
    )

    sock = "/tmp/asi-tpx3-shm.sock"

    conn.serve_shm(sock)
    conn.start_passive()

    while (header := conn.wait_for_arm(timeout=1)) is None:
        print("waiting for header...", header)

    cam_client = CamClient(socket_path=sock)

    while (chunk_stack := conn.get_next_stack(max_size=512*512)) is not None:
        print(chunk_stack)
        chunks = cam_client.get_chunks(handle=chunk_stack)
        print(chunks)

    print(header)
