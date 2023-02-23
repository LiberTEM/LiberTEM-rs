use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use crossbeam::channel::{unbounded, Receiver, Sender, TryRecvError};
use egui::ColorImage;
use log::{debug, error, info};
use ndarray::{s, Array2, ArrayViewMut2};
use websocket::ClientBuilder;

use crate::messages::{AcqMessage, AcquisitionResult, BBox, MessagePart, ProcessingStats};

pub fn decompress_into<T>(
    out_size: [usize; 2],
    data: &[u8],
    dest: &mut ArrayViewMut2<T>,
) -> Option<()> {
    let out_ptr = dest.as_mut_ptr();
    let out_size = out_size[0] * out_size[1];
    match bs_sys::decompress_lz4_into(data, out_ptr, out_size, None) {
        Ok(_) => Some(()),
        Err(e) => {
            error!("decompression failed: {e:?}");
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct AcquisitionData {
    latest: Array2<f32>,
    deltas: Vec<Array2<f32>>,
    bbox: BBox,
    id: String,
}

impl AcquisitionData {
    pub fn new(delta: Array2<f32>, bbox: BBox, id: &str) -> Self {
        Self {
            latest: delta.clone(),
            bbox,
            deltas: vec![delta],
            id: id.to_string(),
        }
    }

    pub fn get_bbox(&self) -> &BBox {
        &self.bbox
    }

    pub fn get_latest(&self) -> &Array2<f32> {
        &self.latest
    }

    pub fn get_id(&self) -> &str {
        &self.id
    }

    pub fn add_delta(&mut self, delta: &Array2<f32>) {
        self.latest = &self.latest + delta;
        self.deltas.push(delta.clone())
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
    to_remove: Vec<String>,
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
    /// the parts into the full results. Returns a vec of acquisition
    /// ids that were changed.
    fn process_pending(&mut self) -> Result<ProcessResults, ParseError> {
        let mut ids: Vec<String> = Vec::new();
        let mut remove_ids: Vec<String> = Vec::new();
        let start_time = Instant::now();

        loop {
            if start_time.elapsed() > Duration::from_millis(1000) {
                break;
            }

            let front_msg = if let Some(msg) = self.pending_messages.pop_front() {
                msg
            } else {
                break;
            };

            let front_clone = front_msg.clone();

            match front_msg {
                MessagePart::AcquisitionStarted(_) => {}
                MessagePart::AcquisitionEnded(ended) => {
                    remove_ids.push(ended.id);
                }
                MessagePart::AcquisitionBinaryPart(_) => {
                    return Err(ParseError::UnexpectedMessage(front_msg.clone()))
                }
                MessagePart::AcquisitionResultHeader(result_header) => {
                    // FIXME: if we have more than one acquisition, how do we handle that?
                    // should we have separate headers?
                    let num_expected = 1;
                    if self.pending_messages.len() < num_expected {
                        self.pending_messages.push_front(front_clone);
                        break;
                    }
                    let msg = self.pending_messages.pop_front().unwrap();
                    if let MessagePart::AcquisitionBinaryPart(bin) = msg {
                        self.merge_result(&bin, &result_header);
                        ids.push(result_header.id.clone());
                    } else {
                        return Err(ParseError::UnexpectedMessage(msg));
                    }
                }
            }
        }

        Ok(ProcessResults {
            processed: ids
                .iter()
                .map(|id| self.acquisitions.get(id).unwrap())
                .collect(),
            to_remove: remove_ids,
        })
    }

    fn merge_result(&mut self, data: &[u8], result: &AcquisitionResult) {
        let key = result.id.clone();
        let mut delta_full = Array2::zeros([result.shape.0 as usize, result.shape.1 as usize]);

        let (ymin, ymax, xmin, xmax) = result.bbox;

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
            .entry(key.to_string())
            .and_modify(|acq_data| {
                acq_data.add_delta(&delta_full);
                acq_data.update_bbox(&bbox);
            })
            .or_insert_with(move || AcquisitionData::new(delta_full, bbox, &key));
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
    let iter_flat = data_flat.indexed_iter();

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

fn receiver_thread(channel: Sender<MessagePart>, stop_event: Arc<AtomicBool>) {
    'outer: loop {
        let mut client = match ClientBuilder::new("ws://localhost:8444")
            .unwrap() // FIXME: parse error in case of bad address; handle once it's user-controlled
            .connect_insecure()
        {
            Ok(client) => client,
            Err(e) => {
                error!("Could not connect: {e}; trying to reconnect...");
                if stop_event.load(Ordering::Relaxed) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(1000));
                continue;
            }
        };

        'inner: loop {
            if stop_event.load(Ordering::Relaxed) {
                break 'outer;
            }

            match MessagePart::from_socket(&mut client) {
                Ok(part) => {
                    debug!("Got a message: {part:?}");
                    channel.send(part).unwrap();
                }
                Err(crate::messages::CommError::Close) => {
                    error!("Connection closed, reconnecting...");
                    break 'inner;
                }
                Err(crate::messages::CommError::NoMessagesAvailable) => {
                    error!("NoMessagesAvailable, reconnecting...");
                    // FIXME: sometimes this means the connection is done...
                    // continue;
                    break 'inner;
                }
                Err(e) => {
                    error!("Got an error while receiving ({e:?}), reconnecting...");
                    break 'inner;
                }
            }
        }
    }
    info!("stopped receiver thread...");
}

pub fn background_thread(sender: Sender<AcqMessage>, stop_event: Arc<AtomicBool>) {
    'outer: loop {
        let (part_sender, part_receiver) = unbounded::<MessagePart>();

        let recv_stop_event = Arc::new(AtomicBool::new(false));

        std::thread::Builder::new()
            .name("receiver".to_string())
            .spawn(move || {
                receiver_thread(part_sender, recv_stop_event);
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
            let process_results = bg_state.process_pending().unwrap();

            for data in process_results.processed {
                let img = render_to_rgb(data.get_latest(), data.get_bbox());
                sender
                    .send(AcqMessage::UpdatedData {
                        id: data.get_id().to_string(),
                        img,
                        bbox: data.get_bbox().to_owned(),
                    })
                    .unwrap();
            }

            sender
                .send(AcqMessage::Stats(ProcessingStats {
                    num_pending,
                    finished: process_results.to_remove.len(),
                }))
                .unwrap();

            for id in process_results.to_remove {
                bg_state.acquisitions.remove(&id);
            }

            let elapsed = start.elapsed();
            let limit = Duration::from_millis(5);

            if elapsed < limit {
                std::thread::sleep(limit - elapsed);
            }
        }
    }

    info!("stopped background thread...");
}
