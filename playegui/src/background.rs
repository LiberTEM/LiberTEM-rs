use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use crossbeam::channel::{unbounded, Receiver, Sender, TryRecvError};
use egui::ColorImage;
use log::{error, info};
use ndarray::{s, Array2, ArrayViewMut2};
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    app::ConnectionStatus,
    messages::{
        AcqMessage, BBox, ChannelDeltaResult, ControlMessage, MessagePart, ProcessingStats,
    },
    receiver::receiver_thread,
};

pub fn decompress_into<T>(
    out_size: [usize; 2],
    data: &[u8],
    dest: &mut ArrayViewMut2<T>,
) -> Option<()> {
    let out_ptr = dest.as_mut_ptr();
    let out_size = out_size[0] * out_size[1];
    match unsafe { bs_sys::decompress_lz4_into(data, out_ptr, out_size, None) } {
        Ok(_) => Some(()),
        Err(e) => {
            error!("decompression failed: {e:?}");
            None
        }
    }
}

/// The results for a single channel of a reconstruction belonging
/// to some acquisition and some UDF.
#[derive(Debug, Clone)]
pub struct ChannelResult {
    latest: Array2<f32>,
    // deltas: Vec<Array2<f32>>,
    bbox: BBox,
    channel_name: String,
    udf_name: String,
}

#[derive(Debug, Clone)]
pub struct AcquisitionData {
    channel_results: HashMap<(String, String), ChannelResult>,
    id: String,
}

impl AcquisitionData {
    fn new(id: String) -> Self {
        Self {
            id,
            channel_results: Default::default(),
        }
    }

    pub fn add_results(
        &mut self,
        udf_name: &str,
        channel_name: &str,
        delta: &Array2<f32>,
        bbox: &BBox,
    ) -> &ChannelResult {
        let key = (udf_name.to_string(), channel_name.to_string());
        self.channel_results
            .entry(key)
            .and_modify(|e| {
                e.add_delta(delta);
                e.update_bbox(bbox);
            })
            .or_insert_with(move || {
                ChannelResult::new(delta.clone(), bbox.clone(), channel_name, udf_name)
            })
    }
}

impl ChannelResult {
    pub fn new(delta: Array2<f32>, bbox: BBox, channel_name: &str, udf_name: &str) -> Self {
        Self {
            latest: delta,
            bbox,
            //deltas: vec![delta],
            channel_name: channel_name.to_string(),
            udf_name: udf_name.to_string(),
        }
    }

    pub fn get_bbox(&self) -> &BBox {
        &self.bbox
    }

    pub fn get_latest(&self) -> &Array2<f32> {
        &self.latest
    }

    pub fn add_delta(&mut self, delta: &Array2<f32>) {
        self.latest = &self.latest + delta;
        // self.deltas.push(delta.clone())
    }

    pub fn update_bbox(&mut self, bbox: &BBox) {
        self.bbox = BBox {
            xmax: self.bbox.xmax.max(bbox.xmax),
            xmin: self.bbox.xmin.min(bbox.xmin),
            ymax: self.bbox.ymax.max(bbox.ymax),
            ymin: self.bbox.ymin.min(bbox.ymin),
        };
    }
}

/// An error happened while consuming the sequence of messages
#[derive(Debug)]
pub enum ParseError {
    /// An unexpected message part was encountered
    UnexpectedMessage(MessagePart),
}

struct BackgroundState {
    acquisitions: HashMap<String, AcquisitionData>,
    pending_messages: VecDeque<MessagePart>,
    receiver: Receiver<MessagePart>,
}

struct ProcessResults<'a> {
    processed: Vec<&'a AcquisitionData>,

    /// IDs of acquisitions that have ended
    ended_ids: Vec<String>,
}

impl BackgroundState {
    fn new(part_receiver: Receiver<MessagePart>) -> Self {
        Self {
            acquisitions: HashMap::new(),
            pending_messages: VecDeque::new(),
            receiver: part_receiver,
        }
    }

    /// Take one or more messages from the pending list and integrate
    /// the parts into the full results. Returns a `ProcessResults`
    /// that contains acquisition data and a list of acquisitions
    /// that have ended.
    fn process_pending(&mut self) -> Result<ProcessResults<'_>, ParseError> {
        let mut ids: Vec<String> = Vec::new();
        let mut ended_ids: Vec<String> = Vec::new();
        let start_time = Instant::now();

        loop {
            if start_time.elapsed() > Duration::from_millis(16) {
                break;
            }

            let front_msg = if let Some(msg) = self.pending_messages.pop_front() {
                msg
            } else {
                break;
            };

            let front_clone = front_msg.clone();

            match front_msg {
                MessagePart::Empty => {}
                MessagePart::AcquisitionStarted(_) => {}
                MessagePart::AcquisitionEnded(ended) => {
                    ended_ids.push(ended.id);
                }
                MessagePart::AcquisitionBinaryPart(_) => {
                    return Err(ParseError::UnexpectedMessage(front_msg.clone()))
                }
                MessagePart::AcquisitionResultHeader(result_header) => {
                    let num_expected = result_header.channels.len();
                    if self.pending_messages.len() < num_expected {
                        self.pending_messages.push_front(front_clone);
                        break;
                    }
                    for chan in result_header.channels.iter() {
                        // won't panic: we checked length before
                        let msg = self.pending_messages.pop_front().unwrap();
                        if let MessagePart::AcquisitionBinaryPart(bin) = msg {
                            if !bin.is_empty() {
                                self.merge_results(&bin, &result_header.id, chan);
                                ids.push(result_header.id.clone());
                            }
                        } else {
                            return Err(ParseError::UnexpectedMessage(msg));
                        }
                    }
                }
                MessagePart::UpdateParams(_) => {} // TODO: send to UI thread
            }
        }

        Ok(ProcessResults {
            processed: ids
                .iter()
                .map(|id| self.acquisitions.get(id).unwrap())
                .collect(),
            ended_ids,
        })
    }

    fn merge_results(
        &mut self,
        data: &[u8],
        acquisition_id: &str,
        channel_result: &ChannelDeltaResult,
    ) {
        let mut delta_full = Array2::zeros([
            channel_result.full_shape.0 as usize,
            channel_result.full_shape.1 as usize,
        ]);

        let (ymin, ymax, xmin, xmax) = channel_result.bbox;

        let bbox = BBox {
            ymin,
            ymax,
            xmin,
            xmax,
        };

        // println!("{:?} {:?}", meta.bbox, bbox);
        let blit_shape = [(ymax - ymin + 1) as usize, (xmax - xmin + 1) as usize];

        let mut delta_blit: Array2<f32> = Array2::zeros(blit_shape);

        decompress_into(blit_shape, data, &mut delta_blit.view_mut());

        let slice_info = s![
            ymin as usize..(ymax + 1) as usize,
            xmin as usize..(xmax + 1) as usize
        ];
        // println!("slice: {:?}", slice_info);
        // println!("delta_blit: {:?}", delta_blit.shape());
        // println!("delta_full: {:?}", delta_full.shape());
        let mut dest = delta_full.slice_mut(slice_info);
        // println!("dest: {:?}", dest.shape());
        dest.assign(&delta_blit);

        self.acquisitions
            .entry(acquisition_id.to_string())
            .and_modify(|acq_data| {
                acq_data.add_results(
                    &channel_result.udf_name,
                    &channel_result.channel_name,
                    &delta_full,
                    &bbox,
                );
            })
            .or_insert_with(move || {
                let mut acq_data = AcquisitionData::new(acquisition_id.to_string());
                acq_data.add_results(
                    &channel_result.udf_name,
                    &channel_result.channel_name,
                    &delta_full,
                    &bbox,
                );
                acq_data
            });
    }

    /// Try to receive some messages, without a set deadline etc.
    fn recv_some_messages(&mut self) {
        while !self.receiver.is_empty() {
            match self.receiver.try_recv() {
                Ok(part) => {
                    self.pending_messages.push_back(part);
                }
                Err(TryRecvError::Disconnected) => panic!("receiver thread is dead"),
                Err(TryRecvError::Empty) => return,
            }
        }
    }
}

pub type ColorTuple = (f32, f32, f32);

fn render_to_rgb(data: &Array2<f32>, bbox: &BBox) -> ColorImage {
    // FIXME: be more intelligent about damage; instead of filtering, use the bboxes!
    let valid = data.iter().filter(|&value| *value != 0.0);
    let (vmin, vmax) = &valid.fold((f32::INFINITY, f32::NEG_INFINITY), |a, &b| {
        (a.0.min(b), a.1.max(b))
    });
    let shape = data.shape();

    let normalizer = |(idx, v): (usize, &f32)| (idx, (v - vmin) / (vmax - vmin));

    let width = data.shape()[1];

    let to_rgba = |(idx, value): (usize, &f32)| {
        let c = 255.0 * *value;

        let x = (idx % width) as u16;
        let y = (idx / width) as u16;

        let a = if x >= bbox.xmin && x <= bbox.xmax && y >= bbox.ymin && y <= bbox.ymax {
            255
        } else {
            0
        };

        [c as u8, c as u8, c as u8, a]
    };

    let flat_shape = data.shape()[0] * data.shape()[1];
    let data_flat = data.to_shape(flat_shape).unwrap();
    let iter_flat = (0..).zip(data_flat.iter());

    let mapped: Vec<u8> = if *vmax == *vmin {
        iter_flat.flat_map(to_rgba).collect()
    } else {
        iter_flat
            .map(normalizer)
            .flat_map(|(idx, v)| to_rgba((idx, &v)))
            .collect()
    };

    ColorImage::from_rgba_unmultiplied([shape[0], shape[1]], &mapped)
}

pub fn background_thread(
    control_channel: Receiver<ControlMessage>,
    acq_message_sender: Sender<AcqMessage>,
    stop_event: Arc<AtomicBool>,
    status: Arc<Mutex<ConnectionStatus>>,
) {
    let recv_stop_event = Arc::new(AtomicBool::new(false));

    'outer: loop {
        let (part_sender, part_receiver) = unbounded::<MessagePart>();

        // some stuff that needs to be moved to the bg thread:
        let recv_stop_event_bg = Arc::clone(&recv_stop_event);
        let recv_control_channel = control_channel.clone();
        let inner_status = Arc::clone(&status);

        std::thread::Builder::new()
            .name("receiver".to_string())
            .spawn(move || {
                receiver_thread(
                    recv_control_channel,
                    part_sender,
                    recv_stop_event_bg,
                    inner_status,
                );
            })
            .unwrap();

        let mut bg_state = BackgroundState::new(part_receiver);

        loop {
            if stop_event.load(Ordering::Relaxed) {
                break 'outer;
            }

            let start = Instant::now();

            bg_state.recv_some_messages();

            let num_pending = bg_state.pending_messages.len();
            // FIXME: reconnect on errors here?
            let process_results = match bg_state.process_pending() {
                Ok(res) => res,
                Err(e) => {
                    error!("error processing pending messages: {:?} reloading...", e);
                    break;
                }
            };

            // for each result and each channel, send an `UpdatedData`
            process_results.processed.par_iter().for_each(|data| {
                for ((_, _), channel_result) in &data.channel_results {
                    let img = render_to_rgb(channel_result.get_latest(), channel_result.get_bbox());
                    acq_message_sender
                        .send(AcqMessage::UpdatedData {
                            id: data.id.to_string(),
                            img,
                            udf_name: channel_result.udf_name.to_string(),
                            channel_name: channel_result.channel_name.to_string(),
                            bbox: channel_result.get_bbox().to_owned(),
                            data: channel_result.get_latest().clone(),
                        })
                        .unwrap();
                }
            });

            acq_message_sender
                .send(AcqMessage::Stats(ProcessingStats {
                    num_pending,
                    finished: process_results.ended_ids.len(),
                }))
                .unwrap();

            for id in process_results.ended_ids {
                bg_state.acquisitions.remove(&id);
                acq_message_sender
                    .send(AcqMessage::AcquisitionEnded(id.clone()))
                    .unwrap();
            }

            let elapsed = start.elapsed();
            let limit = Duration::from_millis(16);

            if elapsed < limit {
                std::thread::sleep(limit - elapsed);
            }
        }
    }

    recv_stop_event.store(true, Ordering::Relaxed);
    info!("stopped background thread...");
}
