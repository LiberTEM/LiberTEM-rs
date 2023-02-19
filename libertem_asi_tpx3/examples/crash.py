import os
import time
import random

import tomli
import scipy.sparse
import numpy as np
from libertem_asi_tpx3 import ASITpx3Connection, CamClient, ChunkStackHandle


def get_reference_array(fn):
    with open(fn, "rb") as f:
        meta = tomli.load(f)

    dirname = os.path.dirname(fn)
    indices = os.path.join(dirname, meta["raw_csr"]["indices_file"])
    indptr = os.path.join(dirname, meta["raw_csr"]["indptr_file"])
    data = os.path.join(dirname, meta["raw_csr"]["data_file"])
    
    indptr_arr = np.memmap(indptr, dtype=meta["raw_csr"]["indptr_dtype"], mode="r")
    indices_arr = np.memmap(indices, dtype=meta["raw_csr"]["indices_dtype"], mode="r")
    data_arr = np.memmap(data, dtype=meta["raw_csr"]["data_dtype"], mode="r")

    arr = scipy.sparse.csr_matrix(
        (data_arr, indices_arr, indptr_arr),
        shape=(int(np.prod(meta["params"]["nav_shape"])), int(np.prod(meta["params"]["sig_shape"]))),
    )
    return arr


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

    for i in range(1000):
        split_at = random.randint(1, 16*512)
        while (header := conn.wait_for_arm(timeout=1)) is None:
            print("waiting for header...")

        print(header)

        cam_client = CamClient(socket_path=sock)

        seen = 0
        frames_cursor = 0

        while (chunk_stack := conn.get_next_stack(max_size=split_at)) is not None:
            cam_client.done(chunk_stack)
