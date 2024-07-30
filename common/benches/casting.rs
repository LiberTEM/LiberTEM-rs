//! Explorative benchmark to check the overhead of different casting methods
//! and different dtype combinations

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use num::{cast::AsPrimitive, NumCast, ToPrimitive};

fn bench_ref_u8(input: &[u8], output: &mut [f32]) {
    output.iter_mut().zip(input.iter()).for_each(|(dest, src)| {
        *dest = *src as f32;
    })
}

fn bench_ref_u16(input: &[u16], output: &mut [f32]) {
    output.iter_mut().zip(input.iter()).for_each(|(dest, src)| {
        *dest = *src as f32;
    })
}

fn bench_ref_u32(input: &[u32], output: &mut [f32]) {
    output.iter_mut().zip(input.iter()).for_each(|(dest, src)| {
        *dest = *src as f32;
    })
}

fn bench_as_primitive<T>(input: &[T], output: &mut [f32])
where
    T: AsPrimitive<f32>,
{
    output.iter_mut().zip(input.iter()).for_each(|(dest, src)| {
        *dest = (*src).as_();
    })
}

fn bench_checked_cast<T>(input: &[T], output: &mut [f32]) -> Option<()>
where
    T: ToPrimitive,
{
    for (dest, src) in output.iter_mut().zip(input.iter()) {
        let converted = (*src).to_f32();
        if let Some(value) = converted {
            *dest = value;
        } else {
            return None;
        }
    }

    Some(())
}

fn bench_checked_cast_to_int<T>(input: &[T], output: &mut [u32]) -> Option<()>
where
    T: ToPrimitive,
{
    for (dest, src) in output.iter_mut().zip(input.iter()) {
        let converted = (*src).to_u32();
        if let Some(value) = converted {
            *dest = value;
        } else {
            return None;
        }
    }

    Some(())
}

fn bench_checked_cast_to_generic<I, O>(input: &[I], output: &mut [O]) -> Option<()>
where
    O: Copy + NumCast,
    I: Copy + ToPrimitive,
{
    for (dest, src) in output.iter_mut().zip(input.iter()) {
        let converted = NumCast::from(*src);
        if let Some(value) = converted {
            *dest = value;
        } else {
            return None;
        }
    }

    Some(())
}

pub fn bench_num_casting(c: &mut Criterion) {
    let input_u8 = (0..(256 * 256))
        .map(|i| (i % 255) as u8)
        .collect::<Vec<u8>>();
    let input_u16 = (0..(256 * 256))
        .map(|e| (e % 65535) as u16)
        .collect::<Vec<u16>>();
    let input_u32 = (0..(256 * 256)).collect::<Vec<u32>>();
    let mut output = Vec::<f32>::new();
    output.resize(input_u16.len(), 0.0);

    let mut output_u32 = vec![0; input_u16.len()];
    //output_u32.resize(input_u16.len(), 0);

    let mut group = c.benchmark_group("cast_to_f32_throughput");

    group.throughput(Throughput::Bytes(
        (input_u16.len() * std::mem::size_of::<f32>()) as u64,
    ));
    // group.throughput(Throughput::Elements((input_u16.len()) as u64));

    group.bench_function("bench_ref_u8", |b| {
        b.iter(|| bench_ref_u8(black_box(&input_u8[..]), black_box(&mut output)))
    });

    group.bench_function("bench_ref_u16", |b| {
        b.iter(|| bench_ref_u16(black_box(&input_u16[..]), black_box(&mut output)))
    });

    group.bench_function("bench_ref_u32", |b| {
        b.iter(|| bench_ref_u32(black_box(&input_u32[..]), black_box(&mut output)))
    });

    group.bench_function("bench_as_primitive-u16", |b| {
        b.iter(|| bench_as_primitive(black_box(&input_u32[..]), black_box(&mut output)))
    });

    group.bench_function("bench_as_primitive-u32", |b| {
        b.iter(|| bench_as_primitive(black_box(&input_u32[..]), black_box(&mut output)))
    });

    group.bench_function("bench_checked_cast-u16", |b| {
        b.iter(|| bench_checked_cast(black_box(&input_u32[..]), black_box(&mut output)))
    });

    group.bench_function("bench_checked_cast-u32", |b| {
        b.iter(|| bench_checked_cast(black_box(&input_u32[..]), black_box(&mut output)))
    });

    group.bench_function("bench_checked_cast_to_int-u16", |b| {
        b.iter(|| bench_checked_cast_to_int(black_box(&input_u32[..]), black_box(&mut output_u32)))
    });

    group.bench_function("bench_checked_cast_to_generic-u16", |b| {
        b.iter(|| {
            bench_checked_cast_to_generic(black_box(&input_u32[..]), black_box(&mut output_u32))
        })
    });

    group.bench_function("bench_checked_cast_to_generic-u16-to-f32", |b| {
        b.iter(|| bench_checked_cast_to_generic(black_box(&input_u32[..]), black_box(&mut output)))
    });

    group.finish();
}

criterion_group!(benches, bench_num_casting);
criterion_main!(benches);
