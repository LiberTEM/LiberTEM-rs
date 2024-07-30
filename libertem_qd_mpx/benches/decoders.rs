use std::fmt::Debug;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::Measurement, BenchmarkGroup,
    Criterion, Throughput,
};
use libertem_qd_mpx::decoder::{RawType, R1, R12, R6};
use num::{cast::AsPrimitive, Bounded, Num, NumCast, ToPrimitive};

fn bench_raw_generic<'a, O, M: Measurement>(
    group: &mut BenchmarkGroup<M>,
    input_u8: &[u8],
    output: &mut [O],
) where
    O: Copy + ToPrimitive + NumCast + Debug + Bounded + Num + 'a + 'static,
    u8: AsPrimitive<O>,
    u16: AsPrimitive<O>,
{
    let mut encoded_r1 = vec![0u8; 256 * 256 / 8];
    R1::encode_all(&mut input_u8.iter(), &mut encoded_r1).unwrap();
    group.bench_function("bench_r1", |b| {
        b.iter(|| R1::decode_all::<_, 64>(black_box(&encoded_r1[..]), black_box(output)))
    });

    let mut encoded_r6 = vec![0u8; 256 * 256];
    R6::encode_all(&mut input_u8.iter(), &mut encoded_r6).unwrap();
    group.bench_function("bench_r6", |b| {
        b.iter(|| R6::decode_all::<_, 8>(black_box(&encoded_r6[..]), black_box(output)))
    });

    let mut encoded_r12 = vec![0u8; 256 * 256 * 2];
    R12::encode_all(&mut input_u8.iter(), &mut encoded_r12).unwrap();
    group.bench_function("bench_r12", |b| {
        b.iter(|| R12::decode_all::<_, 4>(black_box(&encoded_r12[..]), black_box(output)))
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
        b.iter(|| R1::decode_all::<_, 64>(black_box(&encoded_r1[..]), black_box(&mut output)))
    });

    let mut encoded_r6 = vec![0u8; 512 * 512];
    R6::encode_2x2_raw(&input_u8, &mut encoded_r6).unwrap();
    group.bench_function("bench_r6", |b| {
        b.iter(|| R6::decode_all::<_, 8>(black_box(&encoded_r6[..]), black_box(&mut output)))
    });

    let mut encoded_r12 = vec![0u8; 512 * 512 * 2];
    R12::encode_2x2_raw(&input_u8, &mut encoded_r12).unwrap();
    group.bench_function("bench_r12", |b| {
        b.iter(|| R12::decode_all::<_, 4>(black_box(&encoded_r12[..]), black_box(&mut output)))
    });

    group.finish();
}

pub fn bench_decode_chunk(c: &mut Criterion) {
    let input_u8 = [0u8; 8];
    let mut group = c.benchmark_group("bench_decode_chunk");
    group.throughput(Throughput::Bytes(8));

    let mut output = [0u8; 64];
    group.bench_function("bench_r1", |b| {
        b.iter(|| R1::decode_chunk(black_box(&input_u8), black_box(&mut output)))
    });

    let mut output = [0u8; 8];
    group.bench_function("bench_r6", |b| {
        b.iter(|| R6::decode_chunk(black_box(&input_u8), black_box(&mut output)))
    });

    let mut output = [0u8; 4];
    group.bench_function("bench_r12", |b| {
        b.iter(|| R12::decode_chunk(black_box(&input_u8), black_box(&mut output)))
    });
}

fn decode_chunk_r6_ref<'a, OI>(input: &[u8; 8], output: &mut OI)
where
    OI: Iterator<Item = &'a mut u8>,
{
    for (value_chunk, out_value) in input.chunks_exact(2).rev().zip(output) {
        let value = u16::from_be_bytes(value_chunk.try_into().expect("chunked by 2 bytes"));
        *out_value = (value).as_();
    }
}

fn decode_chunk_r6_ref_no_iterator(input: &[u8; 8], output: &mut [u8; 8]) {
    for (value_chunk, out_value) in input.chunks_exact(2).rev().zip(output.iter_mut()) {
        let value = u16::from_be_bytes(value_chunk.try_into().expect("chunked by 2 bytes"));
        *out_value = (value).as_();
    }
}

pub fn bench_decode_chunk_ref(c: &mut Criterion) {
    let input_u8 = [0u8; 8];
    let mut group = c.benchmark_group("bench_decode_chunk_ref");
    group.throughput(Throughput::Bytes(8));

    let mut output = [0u8; 8];
    group.bench_function("bench_r6", |b| {
        b.iter(|| decode_chunk_r6_ref(black_box(&input_u8), black_box(&mut output.iter_mut())))
    });

    group.bench_function("bench_r6_no_iterator", |b| {
        b.iter(|| decode_chunk_r6_ref_no_iterator(black_box(&input_u8), black_box(&mut output)))
    });
}

criterion_group!(
    benches,
    bench_decode_chunk_ref,
    bench_decode_chunk,
    bench_raw,
    bench_raw_quad
);
criterion_main!(benches);
