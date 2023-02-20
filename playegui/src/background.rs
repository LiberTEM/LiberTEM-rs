use std::collections::HashMap;

use colors_transform::Color;
use crossbeam::channel::Sender;
use egui::{TextureHandle, ColorImage, plot::Text};
use log::{error, info, trace};
use ndarray::{Array2, ArrayViewMut2, Array1, s};
use ndarray_stats::QuantileExt;
use scarlet::{colormap::{ColorMap, ListedColorMap}, prelude::{RGBColor}};
use websocket::ClientBuilder;

use crate::messages::{AcqMessage, AcquisitionResult, BBox};

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

struct BackgroundState {
    acquisitions: HashMap<String, AcquisitionData>,
}

impl BackgroundState {
    fn new() -> Self {
        Self {
            acquisitions: HashMap::new(),
        }
    }

    pub fn get_mut(&mut self, acq_id: &str) -> Option<&mut AcquisitionData> {
        self.acquisitions.get_mut(acq_id)
    }

    pub fn get(&self, acq_id: &str) -> Option<&AcquisitionData> {
        self.acquisitions.get(acq_id)
    }

    fn integrate_message(&mut self, msg: AcqMessage) -> Option<&AcquisitionData> {
        match msg {
            AcqMessage::AcquisitionStarted(_) => None,
            AcqMessage::AcquisitionEnded(_) => None,
            AcqMessage::AcquisitionResult(meta, data) => {
                let key = meta.id.clone();
                let mut delta_full = Array2::zeros([meta.shape.0 as usize, meta.shape.1 as usize]);

                let (ymin, ymax, xmin, xmax) = meta.bbox;

                let bbox = BBox {
                    ymin,
                    ymax,
                    xmin,
                    xmax,
                };

                // println!("{:?} {:?}", meta.bbox, bbox);
                let blit_shape = [(ymax - ymin + 1) as usize, (xmax - xmin + 1) as usize];

                let mut delta_blit: Array2<f32> = Array2::zeros(blit_shape);

                decompress_into(blit_shape, &data, &mut delta_blit.view_mut());

                let slice_info = s![ymin as usize..(ymax+1) as usize, xmin as usize..(xmax+1) as usize];
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

                self.acquisitions.get(&meta.id)
            }
            AcqMessage::UpdatedData(..) => {
                None
            },
        }
    }
}

impl Default for BackgroundState {
    fn default() -> Self {
        Self::new()
    }
}

fn render_to_rgb(data: &Array2<f32>) -> ColorImage {
    let valid = data.iter().filter(|&value| *value != 0.0);
    let vmin = &valid.clone().fold(f32::INFINITY, |a, &b| a.min(b));
    let vmax = &valid.fold(0.0f32, |a, &b| a.max(b));
    let shape = data.shape();

    let normalizer = |v: &f32| {
        (v - vmin) / (vmax - vmin)
    };

    let to_rgb = |value: &f32|{
        let hsl = colors_transform::Hsl::from(1.0, 0.0, *value);
        let c = hsl.to_rgb();

        [
            (c.get_red() * 255.0) as u8,
            (c.get_green() * 255.0) as u8,
            (c.get_blue() * 255.0) as u8,
        ]
    };

    let flat_shape = data.shape()[0] * data.shape()[1];
    let data_flat = data.to_shape(flat_shape).unwrap();
    let iter_flat = data_flat.iter();

    let mapped: Vec<u8> = if *vmax == *vmin {
        iter_flat.flat_map(to_rgb).collect()
    } else {
        iter_flat.map(normalizer).flat_map(|v| to_rgb(&v)).collect()
    };

    ColorImage::from_rgb([shape[0], shape[1]], &mapped)
}

fn apply_colormap<CM>(data: &Array2<f32>, colormap: &CM) -> ColorImage
    where
        CM: ColorMap<RGBColor>
{
    let vmin = data.min().unwrap();
    let vmax = data.max().unwrap();
    let shape = data.shape();

    let normalizer = |v: &f32| {
        (v - vmin) / (vmax - vmin)
    };

    let norm_data = if vmax == vmin {
        data.clone()
    } else {
        data.map(normalizer)
    };

    let data_flat: Array1<f32> = norm_data.into_shape(data.shape()[0] * data.shape()[1]).unwrap();

    let rgb: Vec<u8> = colormap.transform(data_flat.iter().map(|value| *value as f64)).iter().flat_map(|c| {
        [
            (c.r * 255.0) as u8,
            (c.g * 255.0) as u8,
            (c.b * 255.0) as u8,
        ]
    }).collect();

    // let rgb: Vec<u8> = data_flat.iter().flat_map(|value| {
    //     let c = colormap.transform_single(*value as f64);
    // }).collect();

    ColorImage::from_rgb([shape[0], shape[1]], &rgb)
}

pub fn background_thread(sender: Sender<AcqMessage>) {
    // FIXME: kill switch for this thread
    loop {
        let mut client = ClientBuilder::new("ws://localhost:8444")
            .unwrap()
            .connect_insecure()
            .unwrap();

        let mut bg_state = BackgroundState::new();
        // let viridis = ListedColorMap::viridis();

        loop {
            // FIXME: error type instead of option so we can reconnect on the right errors
            let msg = AcqMessage::from_socket(&mut client);
            if let Some(msg) = msg {
                trace!("received message: {msg:?}");
                // in case of error, the other thread is dead, so we don't really care...
                if let Some(data) = bg_state.integrate_message(msg.clone()) {
                    let img = render_to_rgb(data.get_latest());
                    sender.send(AcqMessage::UpdatedData(data.get_id().to_string(), img, data.get_bbox().to_owned())).unwrap();
                }
                sender.send(msg).unwrap();
            }
        }
    }
}
