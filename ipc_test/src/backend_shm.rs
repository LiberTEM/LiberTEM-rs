//#![forbid(clippy::unwrap_used)]
//! Raw memory backend using the `shared_memory` crate
use std::{
    fs::{remove_file, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    str::FromStr,
    sync::Mutex,
};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use shared_memory::{Shmem, ShmemConf};

use crate::common::ShmConnectError;

/// Initialization data that we serialize to a file, so our users don't have to
/// pass around so many things out-of-band.
#[derive(Serialize, Deserialize)]
struct InitData<P> {
    size: usize,
    os_handle: String,
    payload: P,
}

pub struct SharedMemory {
    shm_impl: Mutex<Shmem>,
    handle_path: PathBuf,
    is_owner: bool,
}

impl SharedMemory {
    /// Create a new shared memory mapping
    ///
    /// `enable_huge` is not supported and ignored.
    pub fn new<I>(
        _enable_huge: bool,
        handle_path: &Path,
        size: usize,
        init_data: I,
    ) -> Result<Self, ShmConnectError>
    where
        I: Serialize,
    {
        let shm_impl = ShmemConf::new()
            .size(size)
            .create()
            .map_err(|e| ShmConnectError::Other { msg: e.to_string() })?;

        let mut f = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(handle_path)?;

        let init_data_wrapped = InitData {
            size,
            os_handle: shm_impl.get_os_id().to_string(),
            payload: init_data,
        };

        bincode::serialize_into(&f, &init_data_wrapped)?;

        f.flush()?;

        Ok(Self {
            shm_impl: Mutex::new(shm_impl),
            handle_path: handle_path.to_owned(),
            is_owner: true,
        })
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.shm_impl.lock().unwrap().as_ptr()
    }

    pub fn get_handle(&self) -> String {
        self.handle_path.to_str().unwrap().to_owned()
    }

    pub fn connect<I>(handle_path: &str) -> Result<(Self, I), ShmConnectError>
    where
        I: DeserializeOwned,
    {
        let f = OpenOptions::new().read(true).open(handle_path)?;

        let init_data_wrapped: InitData<I> = bincode::deserialize_from(f)?;
        let InitData {
            os_handle,
            size,
            payload,
            ..
        } = init_data_wrapped;

        let shm_impl = ShmemConf::new()
            .os_id(os_handle)
            .size(size)
            .open()
            .map_err(|e| ShmConnectError::Other { msg: e.to_string() })?;

        Ok((
            Self {
                shm_impl: Mutex::new(shm_impl),
                handle_path: PathBuf::from_str(handle_path).unwrap(),
                is_owner: false,
            },
            payload,
        ))
    }
}

impl Drop for SharedMemory {
    fn drop(&mut self) {
        if self.is_owner {
            let _ = remove_file(&self.handle_path);
        }
    }
}

// this is required to be useful via PyO3, without needing to reconnect
// on each shm call
// FIXME: verify that stuff is sufficiently safe!
unsafe impl Send for SharedMemory {}
unsafe impl Sync for SharedMemory {}

pub type Shm = SharedMemory;

#[cfg(test)]
mod tests {
    use crate::{shm::SharedMemory, test_utils::TempDir, SlabInfo};

    #[test]
    fn test_debug_send() {
        fn need_send<S>(_x: S)
        where
            S: Send,
        {
            // nothing in here, just should give us a compilation error
        }

        let handle_path = TempDir::new("handle_path");

        let slab_info: SlabInfo = SlabInfo {
            num_slots: 7,
            slot_size: 1,
            total_size: 7,
        };

        let s: SharedMemory = SharedMemory::new(
            false,
            &handle_path.join("handle.sock"),
            1024 * 1024,
            slab_info,
        )
        .unwrap();

        need_send(s);
    }

    #[test]
    fn test_debug_sync() {
        fn need_sync<S>(_x: S)
        where
            S: Sync,
        {
            // nothing in here, just should give us a compilation error
        }

        let handle_path = TempDir::new("handle_path");

        let slab_info: SlabInfo = SlabInfo {
            num_slots: 7,
            slot_size: 1,
            total_size: 7,
        };

        let s: SharedMemory = SharedMemory::new(
            false,
            &handle_path.join("handle.sock"),
            1024 * 1024,
            slab_info,
        )
        .unwrap();

        need_sync(s);
    }
}
