from contextlib import contextmanager
import os
import logging
import time
from typing import (
    Optional, Tuple, Type, Dict, Any,
)

import numpy as np
from opentelemetry import trace

from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.common.executor import WorkerContext

from libertem.io.dataset.base import (
    DataTile, DataSetMeta, BasePartition, Partition, DataSet, TilingScheme,
)
from libertem.corrections.corrset import CorrectionSet
from libertem_live.detectors.base.controller import (
    AcquisitionController,
)
from libertem_live.detectors.base.acquisition import (
    AcquisitionMixin, AcquisitionProtocol, GetFrames, GenericCommHandler,
)
from libertem_live.detectors.base.connection import (
    PendingAcquisition, DetectorConnection,
)
from libertem_live.hooks import Hooks

from libertem_k2is import (
    K2Connection, K2AcquisitionConfig, K2CamClient, K2Mode, PyAcquisitionSize,
    K2FrameStack,
)

tracer = trace.get_tracer(__name__)
logger = logging.getLogger(__name__)


class K2ISPendingAcquisition(PendingAcquisition):
    def __init__(self):
        super().__init__()

    @property
    def nimages(self) -> int:
        # XXX we don't really know the number of images in the acquisition -
        # unless the user tells us beforehand.
        raise NotImplementedError("how should we know this?")


class K2GetFrames(GetFrames):
    CAM_CLIENT_CLS = K2CamClient
    FRAME_STACK_CLS = K2FrameStack


class K2ISDetectorConnection(DetectorConnection):
    def __init__(
        self,
        local_addr_top: str,
        local_addr_bottom: str,
        sync_mode=None,  # FIXME: immediate or sync to stem?
        camera_mode=K2Mode.IS,
        file_pattern: Optional[str] = None,
        shm_path: Optional[str] = None,
    ):
        self._local_addr_top = local_addr_top
        self._local_addr_bottom = local_addr_bottom
        self._sync_mode = sync_mode
        self._cam_mode = camera_mode
        self._file_pattern = file_pattern
        if shm_path is None:
            shm_path = f"/run/user/{os.getuid()}/k2is-shm-path"
        self._shm_path = shm_path

        self._connect()

    def _connect(self):
        cam = K2Connection(
            local_addr_top=self._local_addr_top,
            local_addr_bottom=self._local_addr_bottom,
            shm_handle_path=self._shm_path,
            frame_stack_size=1,
            mode=self._cam_mode,
            crop_to_image_data=False,
        )
        self._conn = cam

    def wait_for_acquisition(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[PendingAcquisition]:
        # XXX: we don't know how many frames there will be in the acquisition
        # (see `K2ISPendingAcquisition` comment above)
        raise NotImplementedError()

    def get_conn_impl(self):
        return self._conn

    def get_shm_path(self) -> str:
        return self._shm_path

    def close(self):
        self._conn.stop()
        self._conn = None

    def reconnect(self):
        if self._conn is not None:
            self.close()
        self._connect()

    def __enter__(self):
        if self._conn is None:
            self._connect()
        return self

    def get_acquisition_cls(self) -> Type[AcquisitionProtocol]:
        return K2Acquisition


class K2ISConnectionBuilder:
    def open(
        self,
        local_addr_top: str,
        local_addr_bottom: str,
        camera_mode: K2Mode = K2Mode.IS,
        shm_path: Optional[str] = None,
        file_pattern: Optional[str] = None,
    ):
        return K2ISDetectorConnection(
            local_addr_top=local_addr_top,
            local_addr_bottom=local_addr_bottom,
            camera_mode=camera_mode,
            shm_path=shm_path,
            file_pattern=file_pattern,
        )


class K2CommHandler(GenericCommHandler):
    def __init__(self, conn: K2ISDetectorConnection):
        super().__init__(conn=conn)

    def get_conn_impl(self):
        return self._conn.get_conn_impl()

    def start(self):
        pass

    def done(self):
        # continue pumping events for a bit:
        cam = self._conn.get_conn_impl()
        while True:
            frame = cam.get_next_stack(1)
            if frame is None:
                break


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
    frames_per_partition
        A tunable for configuring the feedback rate - more frames per partition
        means slower feedback, possibly less computational overhead, but also
        less parallelism. Might need to be tuned to adapt to the dwell time.
    '''
    def __init__(
        self,
        conn: K2ISDetectorConnection,

        hooks: Optional[Hooks] = None,

        # in passive mode, we get this:
        pending_aq: Optional[K2ISPendingAcquisition] = None,

        controller: Optional[AcquisitionController] = None,

        nav_shape: Optional[Tuple[int, ...]] = None,

        frames_per_partition: Optional[int] = None,
    ):
        super().__init__(
            conn=conn,
            nav_shape=nav_shape,
            frames_per_partition=frames_per_partition,
            controller=controller,
            pending_aq=pending_aq,
            hooks=hooks,
        )
        self._sig_shape: Tuple[int, ...] = ()
        self._frames_per_partition = min(frames_per_partition, prod(nav_shape))

    def initialize(self, executor) -> "DataSet":
        dtype = np.uint16
        conn = self._conn.get_conn_impl()
        shape = conn.get_frame_shape()
        self._sig_shape = shape
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

    def _get_filename(self, pattern: str, values: Dict[str, Any]) -> str:
        assert "%" in pattern
        seq = 1
        all_values = {}
        all_values.update(values)
        all_values['seq'] = seq
        path = pattern % all_values
        while os.path.exists(path):
            seq += 1
            all_values['seq'] = seq
            path = pattern % all_values
        return path

    def start_acquisition(self):
        nimages = prod(self.shape.nav)
        if False:
            file_pattern = self._conn._file_pattern
            if file_pattern is None:
                writer = None
            else:
                filename = self._get_filename(file_pattern, {
                    'nav_shape': "x".join(
                        [str(part) for part in self.shape.nav]
                    ),
                })
                writer = Writer(
                    method="direct",
                    filename=filename,
                )
        conn = self._conn.get_conn_impl()
        conn.start_passive(acquisition_size=PyAcquisitionSize.from_num_frames(nimages))
        conn.wait_for_arm()  # here?

    def end_acquisition(self):
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
                shm_path=self._conn.get_shm_path(),
            )

    def get_task_comm_handler(self) -> "K2CommHandler":
        return K2CommHandler(
            conn=self._conn,
        )


class K2LivePartition(Partition):
    def __init__(
        self, start_idx, end_idx, partition_slice,
        meta, shm_path,
    ):
        super().__init__(
            meta=meta,
            partition_slice=partition_slice,
            io_backend=None,
            decoder=None,
        )
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._shm_path = shm_path

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

    def _get_tiles_fullframe(
        self,
        tiling_scheme: TilingScheme,
        dest_dtype="float32",
        roi=None,
        array_backend=None,
    ):
        assert len(tiling_scheme) == 1
        logger.debug("reading up to frame idx %d for this partition", self._end_idx)
        to_read = self._end_idx - self._start_idx

        with K2GetFrames(
            request_queue=self._worker_context.get_worker_queue(),
            dtype=dest_dtype,
            sig_shape=tuple(tiling_scheme[0].shape),
        ) as frames:
            yield from frames.get_tiles(
                to_read=to_read,
                start_idx=self._start_idx,
                tiling_scheme=tiling_scheme,
                corrections=self._corrections,
                roi=roi,
                array_backend=array_backend,
            )

    def get_tiles(
        self,
        tiling_scheme,
        dest_dtype="float32",
        roi=None,
        array_backend=None,
    ):
        yield from self._get_tiles_fullframe(
            tiling_scheme,
            dest_dtype,
            roi,
            array_backend=array_backend,
        )

    def __repr__(self):
        return f"<K2LivePartition {self._start_idx}:{self._end_idx}>"
