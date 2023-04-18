#![feature(allocator_api)]
// #![feature(adt_const_params)]
#![feature(portable_simd)]

pub mod acquisition;
pub mod allocators;
pub mod args;
pub mod assemble;
pub mod block;
pub mod control;
pub mod decode;
pub mod dio;
pub mod events;
pub mod frame;
pub mod helpers;
pub mod net;
pub mod ordering;
pub mod recv;
pub mod tracing;
pub mod write;
