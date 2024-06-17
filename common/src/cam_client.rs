use ipc_test::{slab::SlabInitError, SharedSlabAllocator};
use ndarray::{ArrayViewMut, ArrayViewMut3, Dimension};

use crate::{
    decoder::{Decoder, DecoderError},
    frame_stack::{FrameMeta, FrameStackHandle},
};

#[derive(thiserror::Error, Debug)]
pub enum CamClientError {
    #[error("failed to connect to {handle_path}: {error}")]
    ConnectError {
        handle_path: String,
        error: SlabInitError,
    },

    #[error("operation on closed client")]
    Closed,

    #[error("handle is already free'd")]
    HandleAlreadyFree,

    #[error("decode failed: {0}")]
    DecodeError(#[from] DecoderError),
}

pub struct GenericCamClient<D>
where
    D: Decoder,
{
    shm: Option<SharedSlabAllocator>,
    decoder: D,
}

/// Client for reading dense data from SHM. That means we get the data as stacks
/// of 2D frames, which either already are strided arrays, or can be decoded
/// into strided arrays.
impl<D> GenericCamClient<D>
where
    D: Decoder,
{
    pub fn new(handle_path: &str) -> Result<Self, CamClientError> {
        match SharedSlabAllocator::connect(handle_path) {
            Ok(shm) => Ok(Self {
                shm: Some(shm),
                decoder: Default::default(),
            }),
            Err(e) => Err(CamClientError::ConnectError {
                handle_path: handle_path.to_owned(),
                error: e,
            }),
        }
    }

    fn get_shm(&self) -> Result<&SharedSlabAllocator, CamClientError> {
        match &self.shm {
            Some(shm) => Ok(shm),
            None => Err(CamClientError::Closed),
        }
    }

    fn get_shm_mut(&mut self) -> Result<&mut SharedSlabAllocator, CamClientError> {
        match &mut self.shm {
            Some(shm) => Ok(shm),
            None => Err(CamClientError::Closed),
        }
    }

    /// Is the data already in a native integer/float format which we can
    /// directly use from numpy?
    ///
    /// In case it is available, use `get_array_zero_copy` to get access to the
    /// array.
    pub fn zero_copy_available<M>(
        &self,
        handle: &FrameStackHandle<M>,
    ) -> Result<bool, CamClientError>
    where
        M: FrameMeta,
    {
        todo!("implement zero_copy_available; delegate to decoder!");
    }

    /// Get an array of the whole frame stack as a strided array.
    ///
    /// This requires that the data is already layed out as a strided array
    /// in the `FrameStackHandle`.
    pub fn get_array_zero_copy<M>(&self, handle: &FrameStackHandle<M>) -> Result<(), CamClientError>
    where
        M: FrameMeta,
    {
        // FIXME: we need to make sure we only loan out the buffer underlying `handle` as long as
        // the `FrameStackHandle` is valid, i.e.
        todo!("implement get_array_zero_copy; delegate to decoder!");
    }

    /// Decode into a pre-allocated array.
    ///
    /// This supports user-allocated memory, which enables things like copying
    /// directly into CUDA locked memory and thus getting rid of a memcpy in the
    /// case of CUDA.
    pub fn decode_into_buffer<M, T>(
        &self,
        input: &FrameStackHandle<M>,
        dest: ArrayViewMut3<'_, T>,
    ) -> Result<(), CamClientError>
    where
        M: FrameMeta,
    {
        self.decode_range_into_buffer(input, dest, 0, input.len() - 1)
    }

    /// Decode a range of frames into a pre-allocated array.
    ///
    /// This allows for decoding only the data that will be processed
    /// immediately afterwards, allowing for more cache-efficient operations.
    pub fn decode_range_into_buffer<M, T>(
        &self,
        input: &FrameStackHandle<M>,
        dest: ArrayViewMut3<'_, T>,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<(), CamClientError>
    where
        M: FrameMeta,
    {
        Ok(self.decoder.decode(input, dest, start_idx, end_idx)?)
    }

    /// Free the given `FrameStackHandle`. When calling this, no Python objects
    /// may have references to the memory of the `handle`.
    pub fn frame_stack_done<M>(&mut self, handle: FrameStackHandle<M>) -> Result<(), CamClientError>
    where
        M: FrameMeta,
    {
        let shm = self.get_shm_mut()?;
        handle.free_slot(shm);
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), CamClientError> {
        self.shm.take();
        Ok(())
    }
}
