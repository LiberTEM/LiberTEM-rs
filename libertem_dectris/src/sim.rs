use std::time::{Duration, Instant};

use log::{debug, info};
use pyo3::{exceptions, prelude::*};
use serde_json::json;
use zmq::{Context, Socket, SocketType::PUSH};

use crate::{
    common::{setup_monitor, DHeader, DetectorConfig, DumpRecordFile, RecordCursor},
    exceptions::TimeoutError,
};

#[derive(Debug)]
pub enum SendError {
    Timeout,
    Other,
}

impl From<zmq::Error> for SendError {
    fn from(e: zmq::Error) -> Self {
        match e {
            zmq::Error::EAGAIN => SendError::Timeout,
            _ => SendError::Other,
        }
    }
}

pub struct FrameSender {
    socket: Socket,
    cursor: RecordCursor,
    detector_config: DetectorConfig,
    series: u64,
    nimages: u64,
    uri: String,
}

impl FrameSender {
    pub fn new(uri: &str, filename: &str, random_port: bool) -> Self {
        let ctx = Context::new();
        let socket = ctx
            .socket(PUSH)
            .expect("context should be able to create a socket");

        if random_port {
            let new_uri = format!("{uri}:*");
            socket.bind(&new_uri).unwrap_or_else(|_| {
                panic!("should be possible to bind the zmq socket at {new_uri}")
            });
        } else {
            socket
                .bind(uri)
                .unwrap_or_else(|_| panic!("should be possible to bind the zmq socket at {uri}"));
        }

        setup_monitor(ctx, "FrameSender".to_string(), &socket);

        let canonical_uri = socket.get_last_endpoint().unwrap().unwrap();

        socket
            .set_sndhwm(4 * 256)
            .expect("should be possible to set sndhwm");

        let file = DumpRecordFile::new(filename);

        // temporary cursor to deserialize headers:
        let mut cursor = file.get_cursor();

        cursor.seek_to_first_header_of_type("dheader-1.0");
        let dheader_raw = cursor.read_raw_msg();
        let dheader: DHeader = serde_json::from_slice(dheader_raw)
            .expect("json should match our serialization schema");

        debug!("{dheader:?}");

        let detector_config: DetectorConfig = cursor.read_and_deserialize().unwrap();
        debug!("{detector_config:?}");

        let nimages = detector_config.get_num_images();
        let series = dheader.series;

        FrameSender {
            socket,
            cursor: file.get_cursor(),
            series,
            nimages,
            detector_config,
            uri: canonical_uri,
        }
    }

    pub fn get_uri(&self) -> &str {
        &self.uri
    }

    pub fn get_detector_config(&self) -> &DetectorConfig {
        &self.detector_config
    }

    pub fn send_frame(&mut self) -> Result<(), SendError> {
        let socket = &self.socket;
        let cursor = &mut self.cursor;

        // We can't just simply blockingly send here, as that will
        // block Ctrl-C when used in Python (the SIGINT handler
        // only sets a flag, it can't interrupt native code)
        // what we can do instead: in "real life", when the high watermark
        // is exceeded, frames are dropped (i.e. "catastrophal" results)
        // So I think it is warranted to go into an error state if the
        // consumer can't keep up.

        // FIXME: We may want to add a "replay speed" later to limit the message
        // rate to something sensible.

        // milliseconds
        socket.set_sndtimeo(1000)?;

        let m = cursor.read_raw_msg();
        socket.send(m, zmq::SNDMORE)?;

        let m = cursor.read_raw_msg();
        socket.send(m, zmq::SNDMORE)?;
        let m = cursor.read_raw_msg();
        socket.send(m, zmq::SNDMORE)?;

        let m = cursor.read_raw_msg();
        socket.send(m, 0)?;

        // back to infinity for the other messages
        // FIXME: might want to have a global timeout later
        // to not have hangs from the Python side in any circumstance
        socket.set_sndtimeo(-1)?;

        Ok(())
    }

    /// Send the message from the current cursor position.
    /// If a timeout occurs, the cursor is rewound to the old
    /// position and a retry can be attempted
    fn send_msg_at_cursor(&mut self) -> Result<(), SendError> {
        let socket = &self.socket;
        let cursor = &mut self.cursor;

        let old_pos = cursor.get_pos();

        let m = cursor.read_raw_msg();
        match socket.send(m, 0) {
            Ok(_) => {}
            Err(zmq::Error::EAGAIN) => {
                cursor.set_pos(old_pos);
                return Err(SendError::Timeout);
            }
            Err(_) => return Err(SendError::Other),
        }

        Ok(())
    }

    fn send_msg_at_cursor_retry<CB>(&mut self, callback: &CB) -> Result<(), SendError>
    where
        CB: Fn() -> Option<()>,
    {
        loop {
            match self.send_msg_at_cursor() {
                Ok(_) => return Ok(()),
                Err(SendError::Timeout) => {
                    if let Some(()) = callback() {
                        continue;
                    } else {
                        return Err(SendError::Timeout);
                    }
                }
                e @ Err(_) => return e,
            }
        }
    }

    pub fn send_headers<CB>(&mut self, idle_callback: CB) -> Result<(), SendError>
    where
        CB: Fn() -> Option<()>,
    {
        // milliseconds
        self.socket.set_sndtimeo(100)?;

        let cursor = &mut self.cursor;
        cursor.seek_to_first_header_of_type("dheader-1.0");

        // dheader
        self.send_msg_at_cursor_retry(&idle_callback)?;

        // detector config
        self.send_msg_at_cursor_retry(&idle_callback)?;

        self.socket.set_sndtimeo(-1)?;

        Ok(())
    }

    pub fn send_frames(&mut self) {
        for _ in 0..self.nimages {
            self.send_frame().expect("send_frame should not time out");
        }
    }

    pub fn send_footer(&mut self) {
        // for simplicity, always "emulate" the footer message
        let footer_json = json!({
            "htype": "dseries_end-1.0",
            "series": self.series,
        });
        self.socket.send(&footer_json.to_string(), 0).unwrap();
    }

    pub fn get_num_frames(&self) -> u64 {
        self.nimages
    }
}

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
        info!("sending {effective_nframes} frames");

        let dwelltime = &slf.dwelltime.clone();
        let sender = &mut slf.frame_sender;

        for frame_idx in 0..effective_nframes {
            py.allow_threads(|| match sender.send_frame() {
                Err(SendError::Timeout) => Err(TimeoutError::new_err(
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
