use std::time::{Duration, Instant};

use pyo3::{exceptions, prelude::*};

use crate::{
    common::{self, DetectorConfig, FrameSender},
    dectris_py::TimeoutError,
};

#[pyclass]
pub struct DectrisSim {
    frame_sender: FrameSender,
    dwelltime: Option<u64>, // in µseconds
}

#[pymethods]
impl DectrisSim {
    #[new]
    fn new(uri: &str, filename: &str, dwelltime: Option<u64>, random_port: bool) -> Self {
        DectrisSim {
            frame_sender: FrameSender::new(uri, filename, random_port),
            dwelltime,
        }
    }

    fn get_uri(slf: PyRef<Self>) -> String {
        slf.frame_sender.get_uri().to_string()
    }

    fn get_detector_config(slf: PyRef<Self>) -> DetectorConfig {
        slf.frame_sender.get_detector_config().clone()
    }

    fn send_headers(mut slf: PyRefMut<Self>, py: Python) -> PyResult<()> {
        let sender = &mut slf.frame_sender;
        py.allow_threads(|| {
            if let Err(e) = sender.send_headers(|| {
                Python::with_gil(|py| {
                    if let Err(e) = py.check_signals() {
                        eprintln!("got python error {e:?}, breaking");
                        None
                    } else {
                        Some(())
                    }
                })
            }) {
                let msg = format!("failed to send headers: {e:?}");
                return Err(exceptions::PyRuntimeError::new_err(msg));
            }
            Ok(())
        })
    }

    /// send `nframes`, if given, or all frames in the acquisition, from the
    /// current position in the file
    fn send_frames(mut slf: PyRefMut<Self>, py: Python, nframes: Option<u64>) -> PyResult<()> {
        let mut t0 = Instant::now();
        let start_time = Instant::now();

        let effective_nframes = match nframes {
            None => slf.frame_sender.get_num_frames(),
            Some(n) => n,
        };

        let dwelltime = &slf.dwelltime.clone();
        let sender = &mut slf.frame_sender;

        for frame_idx in 0..effective_nframes {
            py.allow_threads(|| match sender.send_frame() {
                Err(common::SendError::Timeout) => Err(TimeoutError::new_err(
                    "timeout while sending frames".to_string(),
                )),
                Err(_) => Err(exceptions::PyRuntimeError::new_err(
                    "error while sending frames".to_string(),
                )),
                Ok(_) => Ok(()),
            })?;

            // dwelltime
            // FIXME: for continuous mode, u64 might not be enough for elapsed time,
            // so maybe it's better to carry around a "budget" that can be negative
            // if a frame hasn't been sent out in time etc.
            if let Some(dt) = dwelltime {
                let elapsed_us = start_time.elapsed().as_micros() as u64;
                let target_time_us = (frame_idx + 1) * dt;
                if elapsed_us < target_time_us {
                    let delta = target_time_us - elapsed_us;
                    spin_sleep::sleep(Duration::from_micros(delta));
                }
            }

            // run Python signal handlers every now and then
            if t0.elapsed() > Duration::from_millis(300) {
                t0 = Instant::now();
                py.check_signals()?;

                // also drop GIL once in a while
                py.allow_threads(|| {
                    spin_sleep::sleep(Duration::from_micros(5));
                });
            }
        }

        Ok(())
    }

    fn send_footer(mut slf: PyRefMut<Self>) {
        slf.frame_sender.send_footer();
    }
}
