use std::time::Duration;
use std::time::Instant;

use criterion::BenchmarkId;
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use ipc_test::SharedSlabAllocator;
use k2o::block::K2Block;
use k2o::block_is::K2ISBlock;
use k2o::frame::FrameForWriting;
use k2o::frame::K2Frame;
use k2o::frame_is::K2ISFrame;
use k2o::frame_is::K2ISFrameForWriting;
use tempfile::tempdir;

fn criterion_benchmark(c: &mut Criterion) {
    const PACKET_SIZE: usize = 0x5758;

    const TOTAL_INPUT_SIZE: usize = PACKET_SIZE;

    let block = K2ISBlock::empty(42, 0);

    let socket_dir = tempdir().unwrap();
    let socket_as_path = socket_dir.into_path().join("stuff.socket");

    const FRAME_ID: u32 = 42;
    let mut ssa = SharedSlabAllocator::new(
        10,
        K2ISFrame::FRAME_HEIGHT * K2ISFrame::FRAME_WIDTH * std::mem::size_of::<u16>(),
        false,
        &socket_as_path,
    )
    .expect("create SHM area for testing");
    let mut frame: K2ISFrameForWriting = K2ISFrameForWriting::empty(FRAME_ID, &mut ssa, 0);

    let mut assign_block = c.benchmark_group("assign_block* functions");
    assign_block.measurement_time(Duration::new(10, 0));
    assign_block.throughput(Throughput::Bytes(TOTAL_INPUT_SIZE as u64));
    assign_block.bench_with_input(
        BenchmarkId::new("assign_block", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                frame.assign_block(&block);
            })
        },
    );
    assign_block.finish();
}

// criterion_group!(
//     name = benches;
//     config = Criterion::default().with_measurement(
//         PerfMeasurement::new(PerfMode::Cycles)
//     );
//     targets = criterion_benchmark
// );
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
