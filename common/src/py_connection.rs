#[macro_export]
macro_rules! impl_py_connection {
    (
        $name: ident,
        $frame_stack_type: ident,
        $frame_meta_type: ident,
        $receiver_type: ident,
        $mod: ident
    ) => {
        create_exception!($mod, FrameIteratorError, PyException);

        #[pyclass]
        pub struct $name {
            receiver: $receiver_type,
            remainder: Vec<FrameStackHandle<$frame_meta_type>>,
            shm: SharedSlabAllocator,
            stats: Stats,
        }

        #[pymethods]
        impl $name {
            #[new]
            fn new() -> PyResult<Self> {
                let num_slots = num_slots.map_or_else(|| 2000, |x| x);
                let bytes_per_frame = bytes_per_frame.map_or_else(|| 512 * 512 * 2, |x| x);
                let slot_size = frame_stack_size * bytes_per_frame;
                let shm = match SharedSlabAllocator::new(
                    num_slots,
                    slot_size,
                    huge.map_or_else(|| false, |x| x),
                    &PathBuf::from(handle_path),
                ) {
                    Ok(shm) => shm,
                    Err(e) => {
                        let total_size = num_slots * slot_size;
                        let msg = format!("could not create SHM area (num_slots={num_slots}, slot_size={slot_size} total_size={total_size} huge={huge:?}): {e:?}");
                        return Err(ConnectionError::new_err(msg));
                    }
                };

                let local_shm = shm.clone_and_connect().expect("clone SHM");

                Ok(Self {
                    receiver,
                    remainder: Vec::new(),
                    shm: local_shm,
                    stats: Stats::new(),
                })
            }

            pub fn get_next_stack(
                &mut self,
                max_size: usize,
                py: Python<'_>,
            ) -> PyResult<Option<$frame_stack_type>> {
                let mut iter = $crate::frame_iterator::FrameChunkedIterator::new(
                    &mut self.receiver,
                    &mut self.shm,
                    &mut self.stats,
                );
                match py.allow_threads(|| {
                    iter.get_next_stack_impl(max_size, || {
                        // re-acquire GIL to check if we need to break
                        Python::with_gil(|py| py.check_signals())?;
                        Ok::<_, PyErr>(())
                    })
                }) {
                    Ok(None) => Ok(None),
                    Ok(Some(stack)) => {
                        self.stats.count_stats_item(&stack);
                        Ok(Some($frame_stack_type::new(stack)))
                    },
                    Err(e) => {
                        Err(FrameIteratorError::new_err(e.to_string()))
                    }
                }
            }
        }
    };
}
