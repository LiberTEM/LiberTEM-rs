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
@click.argument('width', type=int, default=512)
@click.argument('height', type=int, default=512)
@click.option('--perf', type=bool, default=False)
def main(width: int, height: int, perf: bool):

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
        nav_shape=(height, width),
        trigger=lambda x: None,
        frames_per_partition=1024,
        controller=conn.get_active_controller(
            trigger_mode='exte',
        )
    ) 
    aq = aq.initialize(ctx.executor)
    print("warmup")
    ctx.run_udf(
        dataset=aq,
        udf=SumSigUDF(),
    )
    if perf:
        with perf_utils.perf('testudf', output_dir='profiles', sample_frequency='max') as perf_data:
            t0 = time.time()
            ctx.run_udf(
                dataset=aq,
                udf=SumSigUDF(),
            )
            t1 = time.time()
            print(t1-t0)
    else:
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
