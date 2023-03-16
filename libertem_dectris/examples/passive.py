import time
import click
import perf_utils
import libertem_dectris
from libertem_live.detectors.dectris.DEigerClient import DEigerClient
from libertem_live.detectors.dectris.acquisition import (
    DectrisAcquisition, DectrisDetectorConnection
)
from libertem_live.api import LiveContext
from libertem.udf.sumsigudf import SumSigUDF


@click.command()
@click.argument('width', type=int, default=256)
@click.argument('height', type=int, default=256)
def main(width: int, height: int):

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

    print("now arm and trigger the detector")
    while (pending_aq := conn.wait_for_acquisition(timeout=10)) is None:
        print("waiting...")

    aq = DectrisAcquisition(
        conn=conn,
        nav_shape=(height, width),
        trigger=lambda x: None,
        frames_per_partition=1024,
        pending_aq=pending_aq,
    ) 
    aq = aq.initialize(ctx.executor)
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


if __name__ == "__main__":
    main()

