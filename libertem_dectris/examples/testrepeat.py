import time
import perf_utils
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient
from libertem_live.detectors.dectris.acquisition import (
    DectrisAcquisition, DectrisDetectorConnection
)
from libertem_live.api import LiveContext
from libertem.udf.sumsigudf import SumSigUDF


if __name__ == "__main__":
    ctx = LiveContext()
    conn = DectrisDetectorConnection(
        api_host='localhost',
        api_port=8910,
        data_host='localhost',
        data_port=9999,
        frame_stack_size=32,
        num_slots=2000,
        bytes_per_frame=512*512,
        huge_pages=True,
    )

    aq = DectrisAcquisition(
        conn=conn,
        nav_shape=(256, 256),
        trigger=lambda x: None,
        frames_per_partition=1024,
        controller=conn.get_active_controller(
            trigger_mode='exte',
        )
    ) 
    aq = aq.initialize(ctx.executor)
    # warmup:
    ctx.run_udf(
        dataset=aq,
        udf=SumSigUDF(),
    )
    for i in range(10):
        t0 = time.time()
        ctx.run_udf(
            dataset=aq,
            udf=SumSigUDF(),
        )
        t1 = time.time()
        print(t1-t0)
        print(f"{aq.shape.nav.size/(t1-t0)}")
    ctx.close()
    conn.close()
    print("done")
