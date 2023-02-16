import time

import numpy as np
from libertem_asi_tpx3 import ASITpx3Connection, CamClient


if __name__ == "__main__":
    conn = ASITpx3Connection(
        uri="localhost:8283",
        chunks_per_stack=16,
        bytes_per_chunk=150000,
        huge=False,
    )

    sock = "/tmp/asi-tpx3-shm.sock"

    conn.serve_shm(sock)
    conn.start_passive()

    for i in range(100):
        while (header := conn.wait_for_arm(timeout=1)) is None:
            print("waiting for header...")

        print(header)

        cam_client = CamClient(socket_path=sock)

        seen = 0

        while (chunk_stack := conn.get_next_stack(max_size=16*512)) is not None:
            # print(chunk_stack)
            chunks = cam_client.get_chunks(handle=chunk_stack)
            for indptr, indices, values in chunks:
                indptr_arr = np.frombuffer(indptr, dtype="uint32")
                indices_arr = np.frombuffer(indices, dtype="uint32")
                values_arr = np.frombuffer(values, dtype="uint32")
                seen += 1

            # print(indptr_arr[0:16])
            # print(indices_arr[0:16])
            # print(values_arr[0:16])

            del indptr_arr
            del indices_arr
            del values_arr
            del indptr
            del indices
            del values
            cam_client.done(chunk_stack)

            # print(f"{len(chunks)} chunks in this stack")

        print(f"chunks seen: {seen}")
        # time.sleep(0.1)
