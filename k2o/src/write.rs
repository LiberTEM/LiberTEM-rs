use core::slice;
use std::{
    fs::{File, OpenOptions},
    os::unix::prelude::FileExt,
};

use hdf5::{
    plist::FileAccess,
    types::{IntSize, TypeDescriptor},
    Dataset, Extent,
};
use memmap2::{MmapMut, MmapOptions};
use ndarray::s;
use ndarray_npy::{write_zeroed_npy, ViewMutNpyExt};

use crate::{
    dio::open_direct,
    events::AcquisitionSize,
    frame::SubFrame,
    helpers::{preallocate, AllocateMode, Shape2, Shape3},
};

#[derive(Debug)]
pub enum WriterError {
    NotResizable,
}

pub trait Writer {
    fn write_frame(&mut self, frame: &SubFrame, frame_idx: u32); // FIXME: possibly return an error here?
    fn resize(&mut self, num_frames: usize) -> Result<(), WriterError>;
}

pub trait WriterBuilder: Send {
    fn open_for_writing(
        &self,
        size: &AcquisitionSize,
        frame_shape: &Shape2,
        pixel_size_bytes: usize,
    ) -> Result<Box<dyn Writer>, WriterError>;

    fn for_filename(filename: &str) -> Box<Self>
    where
        Self: Send + Sized;
}

pub struct DirectWriter {
    file: File,
    filename: String,
    frame_size_bytes: usize,
    pixel_size_bytes: usize,
}

pub struct DirectWriterBuilder {
    filename: String,
}

impl WriterBuilder for DirectWriterBuilder {
    fn open_for_writing(
        &self,
        size: &AcquisitionSize,
        frame_shape: &Shape2,
        pixel_size_bytes: usize,
    ) -> Result<Box<dyn Writer>, WriterError> {
        let mut writer = DirectWriter {
            file: open_direct(&self.filename).expect("could not open file for writing"),
            filename: self.filename.to_string(),
            frame_size_bytes: frame_shape.0 * frame_shape.1 * pixel_size_bytes,
            pixel_size_bytes,
        };
        writer
            .resize(match size {
                AcquisitionSize::NumFrames(frames) => *frames as usize,
                AcquisitionSize::Continuous => {
                    // rely on an outside `resize` call to pre-allocate storage regularly
                    0
                }
            })
            .expect("should be able to resize");
        Ok(Box::new(writer))
    }

    fn for_filename(filename: &str) -> Box<Self>
    where
        Self: Send + Sized,
    {
        Box::new(DirectWriterBuilder {
            filename: filename.to_string(),
        })
    }
}

impl Writer for DirectWriter {
    fn write_frame(&mut self, frame: &SubFrame, frame_idx: u32) {
        let offset = (self.frame_size_bytes * frame_idx as usize) as u64;
        // FIXME: can we get rid of this `unsafe`?
        // https://stackoverflow.com/a/30838655/540644
        // maybe: https://docs.rs/zerocopy/0.6.1/zerocopy/trait.AsBytes.html
        let payload = unsafe {
            let raw_payload = frame.get_payload();
            slice::from_raw_parts(
                raw_payload.as_ptr() as *const u8,
                raw_payload.len() * self.pixel_size_bytes,
            )
        };
        self.file
            .write_all_at(payload, offset)
            .expect("write_all_at should not err");
    }

    fn resize(&mut self, num_frames: usize) -> Result<(), WriterError> {
        preallocate(
            &self.filename,
            self.frame_size_bytes,
            num_frames,
            AllocateMode::AllocateOnly,
        );
        Ok(())
    }
}

pub struct MMapWriterBuilder {
    filename: String,
}

impl WriterBuilder for MMapWriterBuilder {
    fn open_for_writing(
        &self,
        size: &AcquisitionSize,
        frame_shape: &Shape2,
        pixel_size_bytes: usize,
    ) -> Result<Box<dyn Writer>, WriterError> {
        let num_frames = match size {
            AcquisitionSize::NumFrames(frames) => *frames,
            AcquisitionSize::Continuous => 0,
        } as usize;
        {
            let shape: Shape3 = (num_frames, frame_shape.0, frame_shape.1);
            println!("initializing dest file");
            let file = File::create(&self.filename).unwrap();
            write_zeroed_npy::<u16, _>(&file, shape).unwrap();
        }

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.filename)
            .unwrap();
        println!("mapping...");
        let mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
        println!("done mapping.");

        let frame_size_bytes = frame_shape.0 * frame_shape.1 * pixel_size_bytes;

        Ok(Box::new(MMapWriter {
            mmap,
            filename: self.filename.to_string(),
            frame_size_bytes,
        }))
    }

    fn for_filename(filename: &str) -> Box<Self>
    where
        Self: Send + Sized,
    {
        Box::new(MMapWriterBuilder {
            filename: filename.to_string(),
        })
    }
}

/// MMAP writer; writes raw binary files
pub struct MMapWriter {
    mmap: MmapMut,
    filename: String,
    frame_size_bytes: usize,
}

impl MMapWriter {}
impl Writer for MMapWriter {
    fn write_frame(&mut self, frame: &SubFrame, frame_idx: u32) {
        let mut dest_arr = ndarray::ArrayViewMut3::<u16>::view_mut_npy(&mut self.mmap).unwrap();
        let mut dest_slice = dest_arr.slice_mut(s![frame_idx as usize, .., ..]);
        dest_slice.assign(&frame.as_array());
    }

    fn resize(&mut self, num_frames: usize) -> Result<(), WriterError> {
        preallocate(
            &self.filename,
            self.frame_size_bytes,
            num_frames,
            AllocateMode::ZeroFill,
        );
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.filename)
            .unwrap();
        let mmap = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
        self.mmap = mmap;
        Ok(())
    }
}

pub struct HDF5WriterBuilder {
    filename: String,
}

impl WriterBuilder for HDF5WriterBuilder {
    fn open_for_writing(
        &self,
        size: &AcquisitionSize,
        frame_shape: &Shape2,
        _pixel_size_bytes: usize,
    ) -> Result<Box<dyn Writer>, WriterError> {
        match size {
            AcquisitionSize::NumFrames(num_frames) => {
                let mut fa_builder = FileAccess::build();
                fa_builder.fclose_degree(hdf5::file::FileCloseDegree::Weak);
                let fa = fa_builder.finish().unwrap();

                let file = hdf5::File::with_options()
                    .set_access_plist(&fa)
                    .unwrap()
                    .create(&self.filename)
                    .unwrap();

                let shape: Shape3 = (*num_frames as usize, frame_shape.0, frame_shape.1);

                let group = file.create_group("name").unwrap();
                let builder = group.new_dataset_builder();
                let dtype = TypeDescriptor::Unsigned(IntSize::U2);
                let ds = builder
                    .empty_as(&dtype)
                    .shape(shape)
                    .create("data")
                    .unwrap();

                Ok(Box::new(HDF5Writer {
                    ds,
                    is_resizable: false,
                    frame_shape: *frame_shape,
                }))
            }
            AcquisitionSize::Continuous => {
                let mut fa_builder = FileAccess::build();
                fa_builder.fclose_degree(hdf5::file::FileCloseDegree::Weak);
                let fa = fa_builder.finish().unwrap();

                let file = hdf5::File::with_options()
                    .set_access_plist(&fa)
                    .unwrap()
                    .create(&self.filename)
                    .unwrap();

                let group = file.create_group("name").unwrap();
                let builder = group.new_dataset_builder();
                let dtype = TypeDescriptor::Unsigned(IntSize::U2);
                let ds = builder
                    .empty_as(&dtype)
                    //.shape(shape)
                    .shape((Extent::resizable(0), frame_shape.0, frame_shape.1))
                    .create("data")
                    .unwrap();

                Ok(Box::new(HDF5Writer {
                    ds,
                    is_resizable: true,
                    frame_shape: *frame_shape,
                }))
            }
        }
    }

    fn for_filename(filename: &str) -> Box<Self>
    where
        Self: Send + Sized,
    {
        Box::new(HDF5WriterBuilder {
            filename: filename.to_string(),
        })
    }
}

pub struct HDF5Writer {
    ds: Dataset,
    is_resizable: bool,
    frame_shape: Shape2,
}

impl HDF5Writer {}

impl Writer for HDF5Writer {
    fn write_frame(&mut self, frame: &SubFrame, frame_idx: u32) {
        self.ds
            .write_slice(&frame.as_array(), s![frame_idx as usize, .., ..])
            .unwrap();
    }

    fn resize(&mut self, num_frames: usize) -> Result<(), WriterError> {
        if !self.is_resizable {
            return Err(WriterError::NotResizable);
        }
        let shape = (num_frames, self.frame_shape.0, self.frame_shape.1);
        self.ds.resize(shape).unwrap();
        Ok(())
    }
}

pub struct NoopWriter {}

impl Writer for NoopWriter {
    fn write_frame(&mut self, _frame: &SubFrame, _frame_idx: u32) {}

    fn resize(&mut self, _num_frames: usize) -> Result<(), WriterError> {
        Ok(())
    }
}

pub struct NoopWriterBuilder {}

impl NoopWriterBuilder {
    fn new() -> Self {
        Self {}
    }
}

impl Default for NoopWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WriterBuilder for NoopWriterBuilder {
    fn open_for_writing(
        &self,
        _size: &AcquisitionSize,
        _frame_shape: &Shape2,
        _pixel_size_bytes: usize,
    ) -> Result<Box<dyn Writer>, WriterError> {
        Ok(Box::new(NoopWriter {}))
    }

    fn for_filename(_filename: &str) -> Box<Self>
    where
        Self: Send + Sized,
    {
        Box::new(NoopWriterBuilder {})
    }
}
