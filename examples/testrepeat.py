import time
import perf_utils
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient
from libertem_live.detectors.dectris import DectrisAcquisition
from libertem_live.api import LiveContext
from libertem.udf.sumsigudf import SumSigUDF


if __name__ == "__main__":
    ctx = LiveContext()

    aq = DectrisAcquisition(
        api_host='localhost',
        api_port=8910,
        data_host='localhost',
        data_port=9999,
        nav_shape=(256, 256),
        trigger_mode='exte',
        trigger=lambda x: None,
        frames_per_partition=1024,
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
    ctx.close()
    print("done")
