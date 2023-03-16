import os
import time
import tqdm

import tomli
import scipy.sparse
import numpy as np
from libertem_asi_tpx3 import ASITpx3Connection, CamClient, ChunkStackHandle


if __name__ == "__main__":
    sock = "/tmp/asi-tpx3-shm.sock"
    conn = ASITpx3Connection(
        uri="localhost:8283",
        chunks_per_stack=16,
        num_slots=4000,
        bytes_per_chunk=1500000,
        huge=False,
        handle_path=sock,
    )

    conn.start_passive()

    split_at = 16*512
    # split_at = 128

    for i in range(100):
        while (header := conn.wait_for_arm(timeout=1)) is None:
            print("waiting for header...")

        print(header)

        nav_shape = header.get_nav_shape()
        num_frames = nav_shape[0] * nav_shape[1]
        tq = tqdm.tqdm(total=num_frames)

        cam_client = CamClient(handle_path=sock)

        seen = 0
        frames_cursor = 0

        while (chunk_stack := conn.get_next_stack(max_size=split_at)) is not None:
            serialized = chunk_stack.serialize()
            chunk_stack = ChunkStackHandle.deserialize(serialized)

            tq.update(len(chunk_stack))

            # print(chunk_stack)
            chunks = cam_client.get_chunks(handle=chunk_stack)
            for layout, indptr, indices, values in chunks:
                indptr_arr = np.frombuffer(indptr, dtype=layout.get_indptr_dtype())
                indices_arr = np.frombuffer(indices, dtype=layout.get_indices_dtype())
                values_arr = np.frombuffer(values, dtype=layout.get_value_dtype())
                chunk_arr = scipy.sparse.csr_matrix(
                    (values_arr, indices_arr, indptr_arr),
                    shape=(layout.get_nframes(), int(np.prod((516, 516)))),
                )
                seen += 1

                frames_cursor += layout.get_nframes()

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

        tq.close()

        print(f"chunks seen: {seen}")
        # time.sleep(0.1)
