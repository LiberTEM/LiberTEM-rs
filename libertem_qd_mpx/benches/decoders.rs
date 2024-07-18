use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use libertem_qd_mpx::decoder::{RawType, R1, R12, R6};

pub fn bench_raw_formats(c: &mut Criterion) {
    let input_u8 = (0..(256 * 256u32))
        .map(|i| (i % 255) as u8)
        .collect::<Vec<u8>>();
    let input_u16 = (0..(256 * 256))
        .map(|i| (i % 65536) as u16)
        .collect::<Vec<u16>>();
    // let input_u32 = (0..(256 * 256)).collect::<Vec<u32>>();

    let mut output = Vec::<f32>::new();
    output.resize(input_u16.len(), 0.0);

    let mut group = c.benchmark_group("bench_raw_to_f32");

    group.throughput(Throughput::Bytes(
        (input_u16.len() * std::mem::size_of::<f32>()) as u64,
    ));
    // group.throughput(Throughput::Elements((input_u16.len()) as u64));

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

    group.finish();
}

pub fn bench_raw_formats_quad(c: &mut Criterion) {
    let input_u8 = (0..(512 * 512u32))
        .map(|i| (i % 255) as u8)
        .collect::<Vec<u8>>();
    let input_u16 = (0..(512usize * 512usize))
        .map(|i| (i % 2u16.pow(12) as usize) as u16)
        .collect::<Vec<u16>>();
    // let input_u32 = (0..(512 * 512)).collect::<Vec<u32>>();

    let mut output = Vec::<f32>::new();
    output.resize(input_u16.len(), 0.0);

    let mut group = c.benchmark_group("bench_quad_raw_to_f32");

    group.throughput(Throughput::Bytes(
        (input_u16.len() * std::mem::size_of::<f32>()) as u64,
    ));
    // group.throughput(Throughput::Elements((input_u16.len()) as u64));

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

criterion_group!(benches, bench_raw_formats, bench_raw_formats_quad);
criterion_main!(benches);
