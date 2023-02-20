import json
import asyncio

import numpy as np
import bitshuffle
import websockets


async def main():
    async with websockets.connect("ws://localhost:8444") as websocket:
        last_msg = None
        while True:
            msg = await websocket.recv()
            try:
                decoded_msg = json.loads(msg)
                last_msg = decoded_msg
                print(decoded_msg)
            except UnicodeDecodeError as e:
                # binary message, probably
                print(f"binary message of length {len(msg)}")
                if last_msg is not None and last_msg["event"] == "RESULT":
                    # decompress just for fun:
                    decomp = bitshuffle.decompress_lz4(
                        np.frombuffer(msg, dtype="uint8"),
                        dtype=np.dtype(last_msg['dtype']),
                        shape=last_msg['shape']
                    )
                    print(f"decompressed into {decomp.nbytes} bytes")
                else:
                    print(f"last msg: {last_msg}")




if __name__ == "__main__":
    asyncio.run(main())