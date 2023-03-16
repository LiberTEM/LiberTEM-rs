import os
import time
import random

import tqdm
import tomli
import scipy.sparse
import numpy as np
from libertem_asi_tpx3 import ASITpx3Connection, CamClient, ChunkStackHandle


def try_for_split_value(conn, cam_client, split_at):
    print(f"split_at={split_at}")

    for i in tqdm.tqdm(range(64)):
        while (header := conn.wait_for_arm(timeout=1)) is None:
            print("waiting for header...")

        seen = 0
        frames_cursor = 0

        while (chunk_stack := conn.get_next_stack(max_size=split_at)) is not None:
            cam_client.done(chunk_stack)

if __name__ == "__main__":
    sock = "/tmp/asi-tpx3-shm.sock"
    conn = ASITpx3Connection(
        handle_path=sock,
        uri="localhost:8283",
        chunks_per_stack=32,
        bytes_per_chunk=150000,
        num_slots=1000,
        huge=True,
    )

    conn.start_passive()
    cam_client = CamClient(handle_path=sock)

    split_at = None

    try:
        # some specific values:
        try_for_split_value(conn, cam_client, 64)
        try_for_split_value(conn, cam_client, 121)
        try_for_split_value(conn, cam_client, 128)
        try_for_split_value(conn, cam_client, 1024)
        try_for_split_value(conn, cam_client, 4096)

        # brute force for a bit:
        for i in range(1000):
            split_at = random.randint(1, 16*512)
            try_for_split_value(conn, cam_client, split_at)
    except Exception as e:
        raise
