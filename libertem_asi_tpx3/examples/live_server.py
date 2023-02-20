import time
import json
import asyncio
import typing
import copy

import numba
import numpy as np
import uuid
import websockets
from websockets.legacy.server import WebSocketServerProtocol
import lz4.frame
import bitshuffle

from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.executor.pipelined import PipelinedExecutor
from libertem_live.api import LiveContext
from acquisition import AsiAcquisition, AsiDetectorConnection

from libertem.udf.base import UDFResults
from libertem.common.async_utils import sync_to_async

if typing.TYPE_CHECKING:
    from libertem.common.executor import JobExecutor




@numba.njit(cache=True)
def get_bbox(arr) -> typing.Tuple[int, ...]:
    xmin = arr.shape[1]
    ymin = arr.shape[0]
    xmax = 0
    ymax = 0

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            if abs(value) < 1e-8:
                continue
            # got a non-zero value, update indices
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
    return int(ymin), int(ymax), int(xmin), int(xmax)


class WSServer:
    def __init__(self, conn: AsiDetectorConnection, executor: "JobExecutor", ctx: LiveContext):
        self.conn = conn
        self.executor = executor
        self.ctx = ctx
        self.ws_connected = set()
        self.udfs = [
            SumSigUDF(),
        ]

    async def __call__(self, websocket: WebSocketServerProtocol):
        await self.send_state_dump(websocket)
        await self.client_loop(websocket)

    async def send_state_dump(self, websocket):
        pass

    def register_client(self, websocket):
        self.ws_connected.add(websocket)

    def unregister_client(self, websocket):
        self.ws_connected.remove(websocket)

    async def client_loop(self, websocket):
        try:
            self.register_client(websocket)
            async for msg in websocket:
                await self.handle_message(msg, websocket)
        finally:
            self.unregister_client(websocket)

    async def handle_message(self, msg, websocket):
        pass  # TODO: commands from the client

    async def broadcast(self, msg):
        websockets.broadcast(self.ws_connected, msg)

    async def make_delta(self, partial_results: UDFResults, previous_results: typing.Optional[UDFResults]) -> np.ndarray:
        data = partial_results.buffers[0]['intensity'].data
        if previous_results is None:
            data_previous = np.zeros_like(data)
        else:
            data_previous = previous_results.buffers[0]['intensity'].data

        delta = data - data_previous
        return delta

    async def encode_result(self, delta: np.ndarray) -> bytes:
        nonzero_mask = ~np.isclose(0, delta)

        if np.count_nonzero(nonzero_mask) == 0:
            print(f"zero-delta update, skipping")
            # skip this update if it is all-zero
            # TODO: might want to extract delta building into its own function
            # then we can decide outside if we want to run the encoding function or not
            return b"", None  # FIXME: return some structured stuff instead

        bbox = get_bbox(delta)
        ymin, ymax, xmin, xmax = bbox
        delta_for_blit = delta[ymin:ymax + 1, xmin:xmax + 1]
        # print(delta_for_blit.shape, delta_for_blit, list(delta_for_blit[-1]))

        # FIXME: remove allocating copy - maybe copy into pre-allocated buffer instead?
        print(len(delta_for_blit.tobytes()))
        # compressed = await sync_to_async(lambda: lz4.frame.compress(np.copy(delta_for_blit)))
        compressed = await sync_to_async(lambda: bitshuffle.compress_lz4(np.copy(delta_for_blit)))
        print(len(compressed))

        return memoryview(compressed), bbox, delta_for_blit.shape, delta_for_blit.dtype

    async def handle_pending_acquisition(self, pending) -> str:
        acq_id = str(uuid.uuid4())
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_STARTED",
            "id": acq_id,
        }))
        return acq_id

    async def handle_acquisition_end(self, pending, acq_id: str):
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_STARTED",
            "id": acq_id,
        }))

    async def handle_acquisition_end(self, pending, acq_id: str):
        await self.broadcast(json.dumps({
            "event": "ACQUISITION_ENDED",
            "id": acq_id,
        }))

    async def handle_partial_result(
        self,
        partial_results: UDFResults,
        pending_acq,
        acq_id: str,
        previous_results: typing.Optional[UDFResults]
    ):
        delta = await self.make_delta(partial_results, previous_results)
        nonzero_mask = ~np.isclose(0, delta)
        if np.count_nonzero(nonzero_mask) == 0:
            return  # skip this zero-content update

        result_bytes, bbox, shape, dtype = await self.encode_result(delta)
        await self.broadcast(json.dumps({
            "event": "RESULT",
            "bbox": bbox,
            "shape": delta.shape,  # shape of the full delta array, i.e. shape of the result
            "delta_shape": shape,
            "dtype": str(delta.dtype),
            "encoding": "bslz4",
            "id": acq_id,
            # FIXME: "metadata" for this result, like number of buffers, ...
        }))
        # FIXME: might need more than one message!
        await self.broadcast(result_bytes)

    async def serve(self):
        min_delta = 0.01
        async with websockets.serve(self, "localhost", 8444):
            while True:
                pending_acq = await sync_to_async(self.conn.wait_for_acquisition, timeout=10)
                if pending_acq is None:
                    continue
                acq_id = await self.handle_pending_acquisition(pending_acq)
                previous_results = None
                try:
                    aq = self.ctx.prepare_from_pending(
                        pending_acq,
                        conn=conn,
                        pending_aq=pending_acq,
                        frames_per_partition=4*8192,
                    )
                    last_update = 0
                    async for partial_results in ctx.run_udf_iter(dataset=aq, udf=self.udfs, sync=False):
                        if time.time() - last_update > min_delta:
                            await self.handle_partial_result(partial_results, pending_acq, acq_id, previous_results)
                            previous_results = copy.deepcopy(partial_results)
                            last_update = time.time()
                    await self.handle_partial_result(partial_results, pending_acq, acq_id, previous_results)
                    previous_results = copy.deepcopy(partial_results)
                finally:
                    await self.handle_acquisition_end(pending_acq, acq_id)
                previous_results = None



async def main(conn, executor, ctx):
    server = WSServer(conn, executor, ctx)
    await server.serve()


if __name__ == "__main__":
    executor = PipelinedExecutor(
        spec=PipelinedExecutor.make_spec(
            cpus=range(20), cudas=[]
        ),
        pin_workers=False,
        delayed_gc=False,
    )
    ctx = LiveContext(executor=executor)
    conn = AsiDetectorConnection(
        uri="localhost:8283",
        chunks_per_stack=8,
        bytes_per_chunk=1500000,
        num_slots=1000,
    )

    try:
        asyncio.run(main(conn, executor, ctx))
    finally:
        executor.close()
        conn.close()