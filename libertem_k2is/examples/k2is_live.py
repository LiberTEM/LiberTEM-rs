from contextlib import contextmanager
import sys
import logging
import time
from typing import (
    Any, Callable, Generator, Iterable, Iterator, List, Optional,
    Tuple, TypeVar,
)

import numpy as np
from opentelemetry import trace

from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.common.executor import (
    WorkerContext, TaskProtocol, WorkerQueue, TaskCommHandler
)
from libertem.io.dataset.base import (
    DataTile, DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from libertem.corrections.corrset import CorrectionSet
from libertem_live.detectors.base.acquisition import AcquisitionMixin

from k2opy import Cam, Sync, AcquisitionParams, Acquisition, CamClient, Writer


tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


FramesIter = Iterable[Tuple[np.ndarray, int]]


def get_frames(request_queue, socket_path: str) -> FramesIter:
    """
    Consume all FRAMES messages from the request queue until we get an
    END_PARTITION message (which we also consume)
    """
    while True:
        cam_client = CamClient(socket_path)
        with request_queue.get() as msg:
            header, payload_empty = msg
            header_type = header["type"]
            if header_type == "FRAMES":
                idx = header['idx']
                span = trace.get_current_span()
                span.add_event("frame", {"slot": header['slot'], "idx": header["idx"]})
                frame_ref = cam_client.get_frame_ref(header['slot'])
                mv = frame_ref.get_memoryview()
                payload = np.frombuffer(mv, dtype=np.uint16).reshape((1860, 2048))
                yield (payload, idx)
                del payload
                del mv
                del frame_ref
                cam_client.done(header['slot'])
            elif header_type == "END_PARTITION":
                return
            else:
                raise RuntimeError(
                    f"invalid header type {header['type']}; FRAME or END_PARTITION expected"
                )


class K2CommHandler(TaskCommHandler):
    def __init__(self, aq: Acquisition):
        self._aq = aq

    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        aq = self._aq

        with tracer.start_as_current_span("K2CommHandler.handle_task") as span:
            put_time = 0.0
            recv_time = 0.0
            # send the data for this task to the given worker
            partition = task.get_partition()
            slice_ = partition.slice
            start_idx = slice_.origin[0]
            end_idx = slice_.origin[0] + slice_.shape[0]
            span.set_attributes({
                "libertem.partition.start_idx": start_idx,
                "libertem.partition.end_idx": end_idx,
            })
            chunk_size = 64
            current_idx = start_idx
            while current_idx < end_idx:
                t0 = time.perf_counter()
                frame = aq.get_next_frame()
                t1 = time.perf_counter()
                recv_time += t1 - t0
                if frame is None:
                    if current_idx != end_idx:
                        raise RuntimeError("premature end of frame iterator")
                    break

                if frame.is_dropped():
                    span.add_event("dropped frame", {"frame_index": frame.get_idx()})

                assert frame.get_idx() == current_idx, f"{frame.get_idx()} != {current_idx}"

                t0 = time.perf_counter()
                queue.put({
                    "type": "FRAMES",
                    "idx": frame.get_idx(),
                    "slot": aq.get_frame_slot(frame),
                })
                t1 = time.perf_counter()
                put_time += t1 - t0

                current_idx += 1
            span.set_attributes({
                "total_put_time": put_time,
                "total_recv_time": recv_time,
            })

    def start(self):
        socket_path = "/tmp/k2is-socket-todo"
        self._aq.serve_shm(socket_path)

    def done(self):
        pass


class K2Acquisition(AcquisitionMixin, DataSet):
    '''
    Acquisition from a K2IS detector

    Parameters
    ----------

    local_addr_top
        The local IPv4 address where we receive data on (top part)
    local_addr_bottom
        The local IPv4 address where we receive data on (bottom part)

    nav_shape
        The number of scan positions as a 2-tuple :code:`(height, width)`
    sync_mode

    trigger : function
        See :meth:`~libertem_live.api.LiveContext.prepare_acquisition`
        and :ref:`trigger` for details!
    frames_per_partition
        A tunable for configuring the feedback rate - more frames per partition
        means slower feedback, but less computational overhead. Might need to be tuned
        to adapt to the dwell time.
    '''
    def __init__(
        self,
        local_addr_top: str,
        local_addr_bottom: str,
        nav_shape: Tuple[int, ...],
        sync_mode: Sync,
        trigger=lambda aq: None,
        frames_per_partition: int = 128,
    ):
        super().__init__(trigger=trigger)
        try:
            import k2opy  # NOQA
        except ImportError:
            if sys.version_info < (3, 7):
                raise RuntimeError(
                    "K2Acquisition needs at least Python 3.7"
                )
            else:
                raise RuntimeError(
                    "K2Acquisition has additional dependencies; "
                    "please run `pip install libertem-live[k2]` "
                    "to install them."
                )

        self._local_addrs = (local_addr_top, local_addr_bottom)
        self._nav_shape = nav_shape
        self._sig_shape: Tuple[int, ...] = ()
        self._acq_state: Optional[AcquisitionParams] = None
        self._frames_per_partition = min(frames_per_partition, prod(nav_shape))
        self._sync_mode = sync_mode

    def initialize(self, executor) -> "DataSet":
        dtype = np.uint16
        self._sig_shape = (1860, 2048)
        self._meta = DataSetMeta(
            shape=Shape(self._nav_shape + self._sig_shape, sig_dims=2),
            raw_dtype=dtype,
            dtype=dtype,
        )
        return self

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def raw_dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @property
    def meta(self):
        return self._meta

    def get_correction_data(self):
        return CorrectionSet()

    @contextmanager
    def acquire(self):
        with tracer.start_as_current_span('acquire') as span:
            cam = Cam(
                local_addr_top=self._local_addrs[0],
                local_addr_bottom=self._local_addrs[1],
            )
            nimages = prod(self.shape.nav)
            # writer = Writer(
            #     method="direct",
            #     filename="/cachedata/alex/bar.raw",
            # )
            writer = None
            aqp = AcquisitionParams(
                size=nimages,
                sync=self._sync_mode,
                writer=writer,
                enable_frame_iterator=True,
            )
            aq = cam.make_acquisition(aqp)
            try:
                self._acq_state = aq
                aq.arm()

                span.add_event("K2Acquisition.acquire:arm")

                try:
                    # this triggers, either via API or via HW trigger (in which case we
                    # don't need to do anything in the trigger function):
                    with tracer.start_as_current_span("K2Acquisition.trigger"):
                        self.trigger()
                    aq.wait_for_start()
                    t0 = time.time()
                    yield
                finally:
                    try:
                        print(f"acquisition took {time.time() - t0}s")
                    except NameError:
                        pass
                    aq.stop()
                    self._acq_state = None
            finally:
                pass

    def check_valid(self):
        pass

    def need_decode(self, read_dtype, roi, corrections):
        return True  # FIXME: we just do this to get a large tile size

    def adjust_tileshape(self, tileshape, roi):
        depth = 1
        return (depth, *self.meta.shape.sig)

    def get_max_io_size(self):
        # FIXME magic numbers?
        return 12*np.prod(self.meta.shape.sig)*8

    def get_base_shape(self, roi):
        return (1, 1, self.meta.shape.sig[-1])

    @property
    def acquisition_state(self):
        return self._acq_state

    def get_partitions(self):
        num_frames = np.prod(self._nav_shape, dtype=np.uint64)
        num_partitions = int(num_frames // self._frames_per_partition)

        slices = BasePartition.make_slices(self.shape, num_partitions)

        for part_slice, start, stop in slices:
            yield K2LivePartition(
                start_idx=start,
                end_idx=stop,
                meta=self._meta,
                partition_slice=part_slice,
            )

    def get_task_comm_handler(self) -> "K2CommHandler":
        assert self._acq_state is not None
        return K2CommHandler(
            aq=self._acq_state,
        )


class K2LivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice,
        meta,
    ):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=None, decoder=None)
        self._start_idx = start_idx
        self._end_idx = end_idx

    def shape_for_roi(self, roi):
        return self.slice.adjust_for_roi(roi).shape

    @property
    def shape(self):
        return self.slice.shape

    @property
    def dtype(self):
        return self.meta.raw_dtype

    def set_corrections(self, corrections):
        self._corrections = corrections

    def set_worker_context(self, worker_context: WorkerContext):
        self._worker_context = worker_context

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_tiles_fullframe(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None):
        assert len(tiling_scheme) == 1
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx
        depth = tiling_scheme.depth
        buf = np.zeros((depth,) + tiling_scheme[0].shape, dtype=dest_dtype)
        buf_idx = 0
        tile_start = self._start_idx
        socket_path = "/tmp/k2is-socket-todo"
        frames = get_frames(self._worker_context.get_worker_queue(), socket_path)
        while to_read > 0:
            # 1) put frame into tile buffer (including dtype conversion if needed)
            assert buf_idx < depth,\
                    f"buf_idx should be in bounds of buf! ({buf_idx} < ({depth} == {buf.shape[0]}))"
            try:
                frame, abs_idx = next(frames)
                assert frame is not None
                buf[buf_idx] = frame
                # FIXME: "free" the `frame` here
                buf_idx += 1
                to_read -= 1

                # if buf is full, or the partition is done, yield the tile
                tile_done = buf_idx == depth
                partition_done = to_read == 0
            except StopIteration:
                assert to_read == 0, f"we were still expecting to read {to_read} frames more!"
                tile_done = True
                partition_done = True

            if tile_done or partition_done:
                frames_in_tile = buf_idx
                tile_buf = buf[:frames_in_tile]
                if tile_buf.shape[0] == 0:
                    assert to_read == 0
                    continue  # we are done and the buffer is empty

                tile_shape = Shape(
                    (frames_in_tile,) + tuple(tiling_scheme[0].shape),
                    sig_dims=2
                )
                tile_slice = Slice(
                    origin=(tile_start,) + (0, 0),
                    shape=tile_shape,
                )
                # print(f"yielding tile for {tile_slice}")
                self._preprocess(tile_buf, tile_slice)
                yield DataTile(
                    tile_buf,
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )
                tile_start += frames_in_tile
                buf_idx = 0
        logger.debug("LivePartition.get_tiles: end of method")

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        yield from self._get_tiles_fullframe(tiling_scheme, dest_dtype, roi)

    def __repr__(self):
        return f"<K2LivePartition {self._start_idx}:{self._end_idx}>"
