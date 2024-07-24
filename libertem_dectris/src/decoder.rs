use std::any::TypeId;

use common::{
    decoder::{try_cast_if_safe, Decoder, DecoderError, DecoderTargetPixelType},
    frame_stack::FrameStackHandle,
};
use ipc_test::SharedSlabAllocator;

use numpy::ndarray::s;
use zerocopy::{AsBytes, FromBytes};

use crate::base_types::{DectrisFrameMeta, NonEmptyString, PixelType};

#[derive(Debug, Default)]
pub struct DectrisDecoder {}

impl Decoder for DectrisDecoder {
    type FrameMeta = DectrisFrameMeta;

    /// Decode (a part of a) compressed frame stack from the handle `input` to
    /// the array `output`.
    ///
    /// There are ~ three types involved here:
    ///
    /// 1) The output dtype `T`, which is what the user wants to work in (this
    ///    can be an integer type, but also one of the floats, or possibly even
    ///    complex{128,64})
    ///
    /// 2) The "native" output dtype (`DImageD::type_`), which is what we have
    ///    to decompress into (we can't easily map a function over individual
    ///    pixels as part of the decompression) let's call this one `N` (even
    ///    though we don't have it as a concrete type parameter here).  We could
    ///    also call this the intermediate dtype, as it may be different from the
    ///    final output type `T`.
    ///
    /// 3) The per-frame encoding type, like "bs32-lz4<", where the 32 means
    ///    that 32 bits of the input are shuffled together - it doesn't have to
    ///    match `N`! This is mostly an internal encoding type, but we have to be
    ///    sure to handle it independently from the other types.
    ///
    /// The goal is to handle decompression as efficiently as possible, and
    /// especially to handle the "native" case where `T == N` without an extra
    /// copy, but fall back to a almost-as-good method of buffering the data in
    /// as small as possible chunks (sadly in this case: a frame) in the
    /// intermediate dtype, before converting to `T`.
    fn decode<T>(
        &self,
        shm: &SharedSlabAllocator,
        input: &FrameStackHandle<Self::FrameMeta>,
        output: &mut numpy::ndarray::ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), DecoderError>
    where
        T: DecoderTargetPixelType,
    {
        input.with_slot(shm, |slot| {
            // our three cute special cases:
            let mut tmp_u8: Vec<u8> = Vec::new();
            let mut tmp_u16: Vec<u16> = Vec::new();
            let mut tmp_u32: Vec<u32> = Vec::new();

            for ((frame_meta, out_idx), in_idx) in
                input.get_meta().iter().zip(0..).zip(start_idx..end_idx)
            {
                let mut out_view = output.slice_mut(s![out_idx, .., ..]);
                let frame_compressed_data = input.get_slice_for_frame(in_idx, slot);
                let out_slice =
                    out_view
                        .as_slice_mut()
                        .ok_or_else(|| DecoderError::FrameDecodeFailed {
                            msg: "out slice not C-order contiguous".to_owned(),
                        })?;

                let pixel_type = &frame_meta.dimaged.type_;
                let u8_t = TypeId::of::<u8>();
                let u16_t = TypeId::of::<u16>();
                let u32_t = TypeId::of::<u32>();

                let t_type = TypeId::of::<T>();

                if t_type == u8_t || t_type == u16_t || t_type == u32_t {
                    // "zero"-copy shortcut: decompress directly into the destination
                    self.decode_single_frame(
                        frame_compressed_data,
                        out_slice,
                        &frame_meta.dimaged.encoding,
                    )?;
                } else {
                    // in the general case, we need the temporary Vec:
                    let dest_size = frame_meta.get_number_of_pixels();

                    match pixel_type {
                        PixelType::Uint8 => {
                            if tmp_u8.capacity() < dest_size {
                                tmp_u8.resize(dest_size, 0);
                            }
                            self.decode_single_frame(
                                frame_compressed_data,
                                &mut tmp_u8,
                                &frame_meta.dimaged.encoding,
                            )?;
                            try_cast_if_safe(&tmp_u8, out_slice)?;
                        }
                        PixelType::Uint16 => {
                            if tmp_u16.capacity() < dest_size {
                                tmp_u16.resize(dest_size, 0);
                            }
                            self.decode_single_frame(
                                frame_compressed_data,
                                &mut tmp_u16,
                                &frame_meta.dimaged.encoding,
                            )?;
                            try_cast_if_safe(&tmp_u16, out_slice)?;
                        }
                        PixelType::Uint32 => {
                            if tmp_u32.capacity() < dest_size {
                                tmp_u32.resize(dest_size, 0);
                            }
                            self.decode_single_frame(
                                frame_compressed_data,
                                &mut tmp_u32,
                                &frame_meta.dimaged.encoding,
                            )?;
                            try_cast_if_safe(&tmp_u32, out_slice)?;
                        }
                    }
                }
            }

            Ok(())
        })
    }

    fn zero_copy_available(
        &self,
        _handle: &FrameStackHandle<Self::FrameMeta>,
    ) -> Result<bool, DecoderError> {
        Ok(false)
    }
}

impl DectrisDecoder {
    fn decode_single_frame<T: AsBytes + FromBytes>(
        &self,
        input: &[u8],
        output: &mut [T],
        encoding: &NonEmptyString,
    ) -> Result<(), DecoderError> {
        match encoding.as_str() {
            "bs32-lz4<" | "bs16-lz4<" | "bs8-lz4<" => self.decode_single_frame_bslz4(input, output),
            "lz4<" => self.decode_single_frame_plain_lz4(input, output),
            enc => Err(DecoderError::FrameDecodeFailed {
                msg: format!("unknown or unsupported encoding: {enc}"),
            }),
        }
    }

    fn decode_single_frame_bslz4<T>(
        &self,
        input: &[u8],
        output: &mut [T],
    ) -> Result<(), DecoderError> {
        let out_ptr = output.as_mut_ptr();
        unsafe { bs_sys::decompress_lz4_into(&input[12..], out_ptr, output.len(), None) }.map_err(
            |e| {
                let msg = format!("decompression failed: {e:?}");
                DecoderError::FrameDecodeFailed { msg }
            },
        )
    }

    fn decode_single_frame_plain_lz4<T: AsBytes + FromBytes>(
        &self,
        input: &[u8],
        output: &mut [T],
    ) -> Result<(), DecoderError> {
        let out_slice_bytes = output.as_bytes_mut();
        let out_size: i32 = out_slice_bytes.len().try_into().map_err(|e| {
            let msg = format!("output buffer size error: {e}");
            DecoderError::FrameDecodeFailed { msg }
        })?;

        lz4::block::decompress_to_buffer(input, Some(out_size), out_slice_bytes).map_err(|e| {
            let msg = format!("plain lz4 decompression failed: {e}");
            DecoderError::FrameDecodeFailed { msg }
        })?;
        Ok(())
    }
}
