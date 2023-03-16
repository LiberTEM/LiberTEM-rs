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
            decoded_msg = json.loads(msg)
            last_msg = decoded_msg
            print(decoded_msg)

            if decoded_msg['event'] == "RESULT":
                for chan in decoded_msg['channels']:
                    msg = await websocket.recv()
                    print(f"binary message of length {len(msg)}")
                    print(chan)
                    # decompress just for fun:
                    decomp = bitshuffle.decompress_lz4(
                        np.frombuffer(msg, dtype="uint8"),
                        dtype=np.dtype(chan['dtype']),
                        shape=chan['delta_shape']
                    )
                    print(f"decompressed into {decomp.nbytes} bytes")
            else:
                print(f"last msg: {last_msg}")




if __name__ == "__main__":
    asyncio.run(main())
