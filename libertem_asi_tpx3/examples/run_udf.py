import time

from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor
from libertem_live.api import LiveContext
from acquisition import AsiAcquisition, AsiDetectorConnection


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
    n = 1
    try:
        while True:
            print(f"waiting for acquisition {n}")
            pending_acq = conn.wait_for_acquisition(timeout=10)
            if pending_acq is not None:
                print(f"acquisition {n} starting")
                t0 = time.perf_counter()
                aq = ctx.prepare_from_pending(pending_acq, conn=conn, pending_aq=pending_acq, frames_per_partition=4*8192)
                print(pending_acq._acquisition_header)
                udf = SumUDF()
                ctx.run_udf(dataset=aq, udf=udf, plots=False)
                t1 = time.perf_counter()
                print(f"acquisition {n} done in {t1-t0:.3f}s")
                n += 1
    finally:
        executor.close()
        conn.close()
