use std::time::Duration;

use criterion::BenchmarkId;
use criterion::Throughput;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use k2o::block::K2Block;
use k2o::block_is::K2ISBlock;
use k2o::decode::*;

fn criterion_benchmark(c: &mut Criterion) {
    const NUM_SECTORS: usize = 8;
    const BLOCKS_PER_SECTOR_PER_FRAME: usize = 32;
    const NUM_FRAMES: usize = 100;
    const NUM_PACKETS: usize = NUM_FRAMES * NUM_SECTORS * BLOCKS_PER_SECTOR_PER_FRAME;
    const PACKET_SIZE: usize = 0x5758;
    const DECODED_SIZE: usize = (PACKET_SIZE - HEADER_SIZE) * 2 / 3;

    // these are too big for the stack:
    // let input: [u8; PACKET_SIZE * NUM_PACKETS] = [0; PACKET_SIZE * NUM_PACKETS];
    // let mut out: [u16; DECODED_SIZE * NUM_PACKETS] = [0; DECODED_SIZE * NUM_PACKETS];
    // let mut out_f32: [f32; DECODED_SIZE * NUM_PACKETS] = [0.0; DECODED_SIZE * NUM_PACKETS];
    // (in production code, we don't need to make these large allocations, as we only look at ~single packets at a time)

    let input = vec![0 as u8; PACKET_SIZE * NUM_PACKETS].into_boxed_slice();
    let mut out = vec![0 as u16; DECODED_SIZE * NUM_PACKETS].into_boxed_slice();
    let mut out_f32 = vec![0.0 as f32; DECODED_SIZE * NUM_PACKETS].into_boxed_slice();

    const TOTAL_INPUT_SIZE: usize = NUM_PACKETS * PACKET_SIZE;

    let mut group12bit = c.benchmark_group("Decode 12bit");
    group12bit.measurement_time(Duration::new(10, 0));
    group12bit.throughput(Throughput::Bytes(TOTAL_INPUT_SIZE as u64));
    group12bit.bench_with_input(
        BenchmarkId::new("decode", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);
                let out_chunks = out.chunks_exact_mut(DECODED_SIZE);

                for (chunk, o_chunk) in in_chunks.zip(out_chunks) {
                    decode::<PACKET_SIZE>(black_box(&chunk), black_box(o_chunk));
                }
            })
        },
    );

    group12bit.bench_with_input(
        BenchmarkId::new("decode_unrolled", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);
                let out_chunks = out.chunks_exact_mut(DECODED_SIZE);

                for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
                    decode_unrolled::<PACKET_SIZE, DECODED_SIZE>(
                        black_box(&in_chunk),
                        black_box(out_chunk),
                    );
                }
            })
        },
    );

    group12bit.bench_with_input(
        BenchmarkId::new("decode_map_identity", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);
                let out_chunks = out.chunks_exact_mut(DECODED_SIZE);

                for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
                    decode_map::<_, _, PACKET_SIZE, DECODED_SIZE>(
                        black_box(&in_chunk),
                        black_box(out_chunk),
                        |x| x,
                    );
                }
            })
        },
    );

    group12bit.bench_with_input(
        BenchmarkId::new("decode_map_to_f32", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);
                let out_chunks = out_f32.chunks_exact_mut(DECODED_SIZE);

                for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
                    decode_map::<_, _, PACKET_SIZE, DECODED_SIZE>(
                        black_box(&in_chunk),
                        black_box(out_chunk),
                        |x| x as f32,
                    );
                }
            })
        },
    );

    group12bit.bench_with_input(
        BenchmarkId::new("decode_converted_to_f32", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);
                let out_chunks = out_f32.chunks_exact_mut(DECODED_SIZE);

                for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
                    decode_converted::<_, PACKET_SIZE, DECODED_SIZE>(
                        black_box(&in_chunk),
                        black_box(out_chunk),
                    );
                }
            })
        },
    );

    group12bit.bench_with_input(
        BenchmarkId::new("decode_converted_to_u16", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);
                let out_chunks = out.chunks_exact_mut(DECODED_SIZE);

                for (in_chunk, out_chunk) in in_chunks.zip(out_chunks) {
                    decode_converted::<_, PACKET_SIZE, DECODED_SIZE>(
                        black_box(&in_chunk),
                        black_box(out_chunk),
                    );
                }
            })
        },
    );

    group12bit.finish();

    let mut group_from_bytes = c.benchmark_group("K2ISBlock.from_bytes");
    group_from_bytes.measurement_time(Duration::new(10, 0));
    group_from_bytes.throughput(Throughput::Bytes(TOTAL_INPUT_SIZE as u64));

    group_from_bytes.bench_with_input(
        BenchmarkId::new("from_bytes", TOTAL_INPUT_SIZE),
        &TOTAL_INPUT_SIZE,
        |b, &_s| {
            b.iter(|| {
                let in_chunks = input.chunks_exact(PACKET_SIZE);

                for chunk in in_chunks {
                    K2ISBlock::from_bytes(black_box(chunk), 0);
                }
            })
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
