use ipc_test::SharedSlabAllocator;
use log::trace;
use numpy::PyArray3;
use pyo3::{
    exceptions::{self, PyRuntimeError},
    prelude::*,
};
use zerocopy::{AsBytes, FromBytes};

use crate::{
    common::PixelType,
    exceptions::{ConnectionError, DecompressError},
    frame_stack::FrameStackHandle,
};

#[pyclass]
pub struct CamClient {
    shm: Option<SharedSlabAllocator>,
}

impl CamClient {
    fn decompress_bslz4_impl<T: numpy::Element>(
        &self,
        handle: &FrameStackHandle,
        out: &PyArray3<T>,
    ) -> PyResult<()> {
        let mut out_rw = out.readwrite();
        let out_slice = out_rw.as_slice_mut().expect("`out` must be C-contiguous");
        let slot: ipc_test::Slot = if let Some(shm) = &self.shm {
            shm.get(handle.slot.slot_idx)
        } else {
            return Err(PyRuntimeError::new_err("can't decompress with closed SHM"));
        };

        for (frame_meta, idx) in handle.get_meta().iter().zip(0..) {
            let out_size = usize::try_from(frame_meta.get_size()).unwrap();

            // NOTE: frames should all have the same shape
            // FIXME: frames in a stack can _theoretically_ have different bit depth?
            let out_offset = idx * out_size;
            let out_ptr: *mut T = out_slice[out_offset..out_offset + out_size]
                .as_mut_ptr()
                .cast();

            let image_data = handle.get_slice_for_frame(idx, &slot);

            match bs_sys::decompress_lz4_into(&image_data[12..], out_ptr, out_size, None) {
                Ok(()) => {}
                Err(e) => {
                    let msg = format!("decompression failed: {e:?}");
                    return Err(DecompressError::new_err(msg));
                }
            }
        }

        Ok(())
    }

    fn decompress_plain_lz4_impl<T: numpy::Element + AsBytes + FromBytes>(
        &self,
        handle: &FrameStackHandle,
        out: &PyArray3<T>,
    ) -> PyResult<()> {
        let mut out_rw = out.readwrite();
        let out_slice = match out_rw.as_slice_mut() {
            Ok(s) => s,
            Err(e) => {
                let msg = format!("`out` must be C-contiguous: {e:?}");
                return Err(DecompressError::new_err(msg));
            }
        };
        let slot: ipc_test::Slot = if let Some(shm) = &self.shm {
            shm.get(handle.slot.slot_idx)
        } else {
            return Err(PyRuntimeError::new_err("can't decompress with closed SHM"));
        };

        for (frame_meta, idx) in handle.get_meta().iter().zip(0..) {
            // NOTE: frames should all have the same shape
            // FIXME: frames in a stack can _theoretically_ have different bit depth?
            let out_size = usize::try_from(frame_meta.get_size()).unwrap();
            let out_slice_cast = out_slice[0..out_size].as_bytes_mut();
            let image_data = handle.get_slice_for_frame(idx, &slot);

            println!("{} {}", image_data.len(), out_slice_cast.len());
            match lz4::block::decompress_to_buffer(
                image_data,
                Some(out_slice_cast.len().try_into().unwrap()),
                out_slice_cast,
            ) {
                Ok(_) => {}
                Err(e) => {
                    let msg = format!("decompression failed: {e:?}");
                    return Err(DecompressError::new_err(msg));
                }
            }
        }

        Ok(())
    }
}

#[pymethods]
impl CamClient {
    #[new]
    fn new(handle_path: &str) -> PyResult<Self> {
        match SharedSlabAllocator::connect(handle_path) {
            Ok(shm) => Ok(CamClient { shm: Some(shm) }),
            Err(e) => {
                let msg = format!("failed to connect to SHM: {:?}", e);
                Err(ConnectionError::new_err(msg))
            }
        }
    }

    fn decompress_frame_stack(
        slf: PyRef<Self>,
        handle: &FrameStackHandle,
        out: &PyAny,
    ) -> PyResult<()> {
        let arr_u8: Result<&PyArray3<u8>, _> = out.downcast();
        let arr_u16: Result<&PyArray3<u16>, _> = out.downcast();
        let arr_u32: Result<&PyArray3<u32>, _> = out.downcast();

        let (encoding, type_) = if handle.is_empty() {
            return Ok(());
        } else {
            let dimaged = &handle.get_meta().first().unwrap().dimaged;
            (&dimaged.encoding, &dimaged.type_)
        };

        match encoding.as_str() {
            // "<encoding>: String of the form ”[bs<BIT>][[-]lz4][<|>]”. bs<BIT> stands for bit shuffling with <BIT> bits, lz4 for
            // lz4 compression and < (>) for little (big) endian. E.g. ”bs8-lz4<” stands for 8bit bitshuffling, lz4 compression
            // and little endian. lz4 data is written as defined at https://code.google.com/p/lz4/ without any additional data like
            // block size etc."
            "bs32-lz4<" | "bs16-lz4<" | "bs8-lz4<" => match type_ {
                PixelType::Uint8 => slf.decompress_bslz4_impl(handle, arr_u8.unwrap())?,
                PixelType::Uint16 => slf.decompress_bslz4_impl(handle, arr_u16.unwrap())?,
                PixelType::Uint32 => slf.decompress_bslz4_impl(handle, arr_u32.unwrap())?,
            },
            "lz4<" => match type_ {
                PixelType::Uint8 => slf.decompress_plain_lz4_impl(handle, arr_u8.unwrap())?,
                PixelType::Uint16 => slf.decompress_plain_lz4_impl(handle, arr_u16.unwrap())?,
                PixelType::Uint32 => slf.decompress_plain_lz4_impl(handle, arr_u32.unwrap())?,
            },
            e => {
                let msg = format!("can't deal with encoding {e}");
                return Err(exceptions::PyValueError::new_err(msg));
            }
        }
        Ok(())
    }

    fn done(mut slf: PyRefMut<Self>, handle: &FrameStackHandle) -> PyResult<()> {
        let slot_idx = handle.slot.slot_idx;
        if let Some(shm) = &mut slf.shm {
            shm.free_idx(slot_idx);
            Ok(())
        } else {
            Err(PyRuntimeError::new_err(
                "CamClient.done called with SHM closed",
            ))
        }
    }

    fn close(&mut self) {
        self.shm.take();
    }
}

impl Drop for CamClient {
    fn drop(&mut self) {
        trace!("CamClient::drop");
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use lz4::block::CompressionMode;
    use numpy::PyArray;
    use tempfile::tempdir;

    use ipc_test::SharedSlabAllocator;
    use pyo3::{prepare_freethreaded_python, Python};
    use zerocopy::AsBytes;

    use crate::{
        cam_client::CamClient,
        frame_stack::{FrameStackForWriting, FrameStackHandle},
    };
    use tempfile::TempDir;

    fn get_socket_path() -> (TempDir, PathBuf) {
        let socket_dir = tempdir().unwrap();
        let socket_as_path = socket_dir.path().join("stuff.socket");

        (socket_dir, socket_as_path)
    }

    #[test]
    fn test_cam_client() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, 512);
        let dimage = crate::common::DImage {
            htype: "".to_string(),
            series: 1,
            frame: 1,
            hash: "".to_string(),
        };
        let dimaged = crate::common::DImageD {
            htype: "".to_string(),
            shape: vec![16, 16],
            type_: crate::common::PixelType::Uint16,
            encoding: "bs16-lz4<".to_string(),
        };
        let dconfig = crate::common::DConfig {
            htype: "".to_string(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };

        // some predictable test data:
        let in_: Vec<u16> = (0..256).map(|i| i % 16).collect();
        let compressed_data = bs_sys::compress_lz4(&in_, None).unwrap();

        // compressed dectris data stream has an (unknown)
        // header in front of the compressed data, which we just cut off,
        // so here we just prepend 12 zero-bytes
        let mut data_with_prefix = vec![0; 12];
        data_with_prefix.extend_from_slice(&compressed_data);
        assert!(data_with_prefix.len() < 512);
        data_with_prefix.iter().take(12).for_each(|&e| {
            assert_eq!(e, 0);
        });
        println!("{:x?}", &compressed_data);
        println!("{:x?}", &data_with_prefix[12..]);
        assert_eq!(fs.cursor, 0);
        fs.frame_done(dimage, dimaged, dconfig, &data_with_prefix);
        assert_eq!(fs.cursor, data_with_prefix.len());

        // we have one frame in there:
        assert_eq!(fs.len(), 1);

        let fs_handle = fs.writing_done(&mut shm);

        // we still have one frame in there:
        assert_eq!(fs_handle.len(), 1);

        // initialize a Python interpreter so we are able to construct a PyBytes instance:
        prepare_freethreaded_python();

        // roundtrip serialize/deserialize:
        Python::with_gil(|py| {
            let bytes = fs_handle.serialize(py).unwrap();
            let new_handle = FrameStackHandle::deserialize_impl(bytes.as_ref(py)).unwrap();
            assert_eq!(fs_handle, new_handle);
        });

        let client = CamClient::new(socket_as_path.to_str().unwrap()).unwrap();

        let slot_r: ipc_test::Slot = shm.get(fs_handle.slot.slot_idx);
        let slice = slot_r.as_slice();
        println!("{:x?}", slice);

        Python::with_gil(|py| {
            let flat: Vec<u16> = (0..256).collect();
            let out = PyArray::from_vec(py, flat).reshape((1, 16, 16)).unwrap();
            client.decompress_bslz4_impl(&fs_handle, out).unwrap();

            out.readonly()
                .as_slice()
                .unwrap()
                .iter()
                .zip(0..)
                .for_each(|(&item, idx)| {
                    assert_eq!(item, in_[idx]);
                    assert_eq!(item, (idx % 16) as u16);
                });
        });
    }

    #[test]
    fn test_cam_client_lz4() {
        let (_socket_dir, socket_as_path) = get_socket_path();
        let mut shm = SharedSlabAllocator::new(1, 4096, false, &socket_as_path).unwrap();
        let slot = shm.get_mut().expect("get a free shm slot");
        let mut fs = FrameStackForWriting::new(slot, 1, 512);
        let dimage = crate::common::DImage {
            htype: "".to_string(),
            series: 1,
            frame: 1,
            hash: "".to_string(),
        };
        let dimaged = crate::common::DImageD {
            htype: "".to_string(),
            shape: vec![16, 16],
            type_: crate::common::PixelType::Uint16,
            encoding: "lz4<".to_string(),
        };
        let dconfig = crate::common::DConfig {
            htype: "".to_string(),
            start_time: 0,
            stop_time: 0,
            real_time: 0,
        };

        // some predictable test data:
        let in_: Vec<u16> = (0..256).map(|i| i % 16).collect();
        let in_bytes = in_.as_bytes();
        let compressed_data =
            lz4::block::compress(in_bytes, Some(CompressionMode::DEFAULT), false).unwrap();

        println!("{:x?}", &compressed_data);
        assert_eq!(fs.cursor, 0);
        fs.frame_done(dimage, dimaged, dconfig, &compressed_data);
        assert_eq!(fs.cursor, compressed_data.len());

        // we have one frame in there:
        assert_eq!(fs.len(), 1);

        let fs_handle = fs.writing_done(&mut shm);

        // we still have one frame in there:
        assert_eq!(fs_handle.len(), 1);

        // initialize a Python interpreter so we are able to construct a PyBytes instance:
        prepare_freethreaded_python();

        // roundtrip serialize/deserialize:
        Python::with_gil(|py| {
            let bytes = fs_handle.serialize(py).unwrap();
            let new_handle = FrameStackHandle::deserialize_impl(bytes.as_ref(py)).unwrap();
            assert_eq!(fs_handle, new_handle);
        });

        let client = CamClient::new(socket_as_path.to_str().unwrap()).unwrap();

        let slot_r: ipc_test::Slot = shm.get(fs_handle.slot.slot_idx);
        let slice = slot_r.as_slice();
        let slice_for_frame = fs_handle.get_slice_for_frame(0, &slot_r);

        // try decompression directly:
        let out_size = 256 * TryInto::<i32>::try_into(std::mem::size_of::<u16>()).unwrap();
        println!(
            "slice_for_frame.len(): {}, uncompressed_size: {}",
            slice_for_frame.len(),
            out_size
        );
        lz4::block::decompress(slice_for_frame, Some(out_size)).unwrap();

        println!("{:x?}", slice_for_frame);
        println!("{:x?}", slice);

        Python::with_gil(|py| {
            let flat: Vec<u16> = (0..256).collect();
            let out = PyArray::from_vec(py, flat).reshape((1, 16, 16)).unwrap();

            client.decompress_plain_lz4_impl(&fs_handle, out).unwrap();

            out.readonly()
                .as_slice()
                .unwrap()
                .iter()
                .zip(0..)
                .for_each(|(&item, idx)| {
                    assert_eq!(item, in_[idx]);
                    assert_eq!(item, (idx % 16) as u16);
                });
        });
    }
}
