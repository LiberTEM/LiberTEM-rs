use std::fmt::Debug;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    Criterion, Throughput,
};
use libertem_qd_mpx::decoder::{RawType, R1, R12, R6};
use num::{NumCast, ToPrimitive};

fn bench_raw_generic<'a, O, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    input_u8: &[u8],
    output: &mut [O],
) where
    O: Copy + ToPrimitive + NumCast + Debug + 'a,
{
    let mut encoded_r1 = vec![0u8; 256 * 256 / 8];
    R1::encode_all(&mut input_u8.iter(), &mut encoded_r1).unwrap();
    group.bench_function("bench_r1", |b| {
        b.iter(|| {
            R1::decode_all(
                black_box(&encoded_r1[..]),
                black_box(&mut output.iter_mut()),
            )
        })
    });

    let mut encoded_r6 = vec![0u8; 256 * 256];
    R6::encode_all(&mut input_u8.iter(), &mut encoded_r6).unwrap();
    group.bench_function("bench_r6", |b| {
        b.iter(|| {
            R6::decode_all(
                black_box(&encoded_r6[..]),
                black_box(&mut output.iter_mut()),
            )
        })
    });

    let mut encoded_r12 = vec![0u8; 256 * 256 * 2];
    R12::encode_all(&mut input_u8.iter(), &mut encoded_r12).unwrap();
    group.bench_function("bench_r12", |b| {
        b.iter(|| {
            R12::decode_all(
                black_box(&encoded_r12[..]),
                black_box(&mut output.iter_mut()),
            )
        })
    });
}

pub fn bench_raw(c: &mut Criterion) {
    let input_u8 = (0..(256 * 256u32))
        .map(|i| (i % 255) as u8)
        .collect::<Vec<u8>>();

    let mut output = vec![0u8; input_u8.len()];
    let mut group = c.benchmark_group("bench_raw_to_u8");
    group.throughput(Throughput::Bytes(input_u8.len() as u64));
    bench_raw_generic(&mut group, &input_u8, &mut output);
    group.finish();

    let mut output = vec![0u16; input_u8.len()];
    let mut group = c.benchmark_group("bench_raw_to_u16");
    group.throughput(Throughput::Bytes(input_u8.len() as u64));
    bench_raw_generic(&mut group, &input_u8, &mut output);
    group.finish();

    let mut output = vec![0u32; input_u8.len()];
    let mut group = c.benchmark_group("bench_raw_to_u32");
    group.throughput(Throughput::Bytes(input_u8.len() as u64));
    bench_raw_generic(&mut group, &input_u8, &mut output);
    group.finish();

    let mut output = vec![0f32; input_u8.len()];
    let mut group = c.benchmark_group("bench_raw_to_f32");
    group.throughput(Throughput::Bytes(input_u8.len() as u64));
    bench_raw_generic(&mut group, &input_u8, &mut output);
    group.finish();
}

pub fn bench_raw_quad(c: &mut Criterion) {
    let input_u8 = (0..(512 * 512u32))
        .map(|i| (i % 255) as u8)
        .collect::<Vec<u8>>();

    let mut output = Vec::<f32>::new();
    output.resize(input_u8.len(), 0.0);

    let mut group = c.benchmark_group("bench_quad_raw_to_f32");
    group.throughput(Throughput::Bytes(input_u8.len() as u64));

    let mut encoded_r1 = vec![0u8; 512 * 512 / 8];
    R1::encode_2x2_raw(&input_u8, &mut encoded_r1).unwrap();
    group.bench_function("bench_r1", |b| {
        b.iter(|| {
            R1::decode_all(
                black_box(&encoded_r1[..]),
                black_box(&mut output.iter_mut()),
            )
        })
    });

    let mut encoded_r6 = vec![0u8; 512 * 512];
    R6::encode_2x2_raw(&input_u8, &mut encoded_r6).unwrap();
    group.bench_function("bench_r6", |b| {
        b.iter(|| {
            R6::decode_all(
                black_box(&encoded_r6[..]),
                black_box(&mut output.iter_mut()),
            )
        })
    });

    let mut encoded_r12 = vec![0u8; 512 * 512 * 2];
    R12::encode_2x2_raw(&input_u8, &mut encoded_r12).unwrap();
    group.bench_function("bench_r12", |b| {
        b.iter(|| {
            R12::decode_all(
                black_box(&encoded_r12[..]),
                black_box(&mut output.iter_mut()),
            )
        })
    });

    group.finish();
}

criterion_group!(benches, bench_raw, bench_raw_quad);
criterion_main!(benches);
