import os
import time

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
    sock = "/tmp/asi-tpx3-shm.sock"
    conn = ASITpx3Connection(
        uri="localhost:8283",
        handle_path=sock,
        chunks_per_stack=2,
        num_slots=4000,
        bytes_per_chunk=150000,
        huge=False,
    )

    conn.start_passive()

    reference_array = get_reference_array("/cachedata/alex/tpx3/csr_streaming/sparse.toml")
    print(reference_array)

    split_at = 16*512
    # split_at = 128

    for i in range(10):
        while (header := conn.wait_for_arm(timeout=1)) is None:
            print("waiting for header...")

        print(header)

        cam_client = CamClient(handle_path=sock)

        seen = 0
        frames_cursor = 0

        while (chunk_stack := conn.get_next_stack(max_size=split_at)) is not None:
            serialized = chunk_stack.serialize()
            chunk_stack = ChunkStackHandle.deserialize(serialized)

            # print(chunk_stack)
            chunks = cam_client.get_chunks(handle=chunk_stack)
            for layout, indptr, indices, values in chunks:
                # print(layout)
                indptr_arr = np.frombuffer(indptr, dtype=layout.get_indptr_dtype())
                indices_arr = np.frombuffer(indices, dtype=layout.get_indices_dtype())
                values_arr = np.frombuffer(values, dtype=layout.get_value_dtype())
                chunk_arr = scipy.sparse.csr_matrix(
                    (values_arr, indices_arr, indptr_arr),
                    shape=(layout.get_nframes(), int(np.prod((516, 516)))),
                )
                seen += 1

                ref_slice = reference_array[frames_cursor:frames_cursor + layout.get_nframes(), :]
                assert ref_slice.nnz == chunk_arr.nnz
                assert np.allclose(ref_slice.data, chunk_arr.data)
                assert np.allclose(ref_slice.indices, chunk_arr.indices)
                assert np.allclose(ref_slice.indptr, chunk_arr.indptr)

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

        print(f"chunks seen: {seen}")
        # time.sleep(0.1)
