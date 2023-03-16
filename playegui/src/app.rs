use std::{
    collections::HashMap,
    f64::consts::PI,
    sync::{
        atomic::{AtomicBool, Ordering::Relaxed},
        Arc, Mutex,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

use crossbeam::channel::{unbounded, Receiver, Sender};
use eframe::epaint::ahash::HashSet;
use egui::{
    plot::{Corner, HLine, Legend, MarkerShape, Plot, PlotImage, PlotPoint, Points, Polygon},
    vec2, ColorImage, TextureHandle, TextureOptions, Ui,
};
use egui_extras::RetainedImage;
use log::trace;
use ndarray::Array2;

use crate::{
    background::background_thread,
    messages::{
        AcqMessage, BBox, ControlMessage, ProcessingStats, UpdateParams, UpdateParamsInner,
    },
};

#[derive(Debug)]
pub enum ConnectionStatus {
    Connecting,
    Connected,
    Error,
}

/// Images need to be uploaded and become textures, textures can be used
/// directly
enum ImageOrTexture {
    Image(ColorImage),
    Texture(TextureHandle),
}

#[derive(Clone)]
struct AggregateStats {
    ts: Instant,
    num_pending: usize,
    finished: usize,
}

impl AggregateStats {
    fn new() -> Self {
        Self {
            ts: Instant::now(),
            num_pending: 0,
            finished: 0,
        }
    }

    fn with_updated_ts(self) -> Self {
        AggregateStats {
            ts: Instant::now(),
            num_pending: self.num_pending,
            finished: self.finished,
        }
    }

    fn aggregate(&mut self, stats: &ProcessingStats) {
        self.num_pending = stats.num_pending;
        self.finished += stats.finished;
    }
}

#[derive(Hash, Debug, Clone, Eq, PartialEq)]
struct ResultId {
    acquisition: String,
    udf_name: String,
    channel_name: String,
}

#[derive(Debug, Clone)]
struct RingParams {
    pub cx: f32,
    pub cy: f32,
    pub ri: f32,
    pub ro: f32,
}

impl Default for RingParams {
    fn default() -> Self {
        Self {
            cx: 260.0,
            cy: 250.0,
            ri: 140.0,
            ro: 200.0,
        }
    }
}

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
// #[derive(serde::Deserialize, serde::Serialize)]
// #[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    ring_params: RingParams,
    data_source: Receiver<AcqMessage>,
    control_channel: Sender<ControlMessage>,
    acquisitions: HashMap<ResultId, (ImageOrTexture, BBox, Array2<f32>)>,
    acquisition_history: Vec<ResultId>,
    stop_event: Arc<AtomicBool>,
    bg_thread: Option<JoinHandle<()>>, // Option<> so we can `take` it out when joining
    stats: AggregateStats,
    previous_stats: Option<(AggregateStats, AggregateStats)>,
    conn_status: Arc<Mutex<ConnectionStatus>>,
    logo: RetainedImage,
}

fn circle_points(cx: f64, cy: f64, r: f64) -> Vec<[f64; 2]> {
    const N: usize = 100;

    // sample from 0 to 2PI in N steps:
    (0..N)
        .map(|step| {
            let step = step as f64;
            let step_ratio = step / (N as f64);
            let rad = 2.0 * PI * step_ratio;
            let x = cx + r * rad.cos();
            let y = cy + r * rad.sin();
            [x, y]
        })
        .collect()
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        let stop_event = Arc::new(AtomicBool::new(false));

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        // if let Some(storage) = cc.storage {
        //     let mut loaded: TemplateApp =
        //         eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
        //     let (join_handle, data_source, control_channel) = Self::connect(Arc::clone(&stop_event));
        //     loaded.data_source = Some(data_source);
        //     loaded.bg_thread = Some(join_handle);
        //     loaded.control_channel = Some(control_channel);
        //     return loaded;
        // }

        let conn_status = Arc::new(Mutex::new(ConnectionStatus::Connecting));

        let (join_handle, data_source, control_channel) =
            Self::connect(Arc::clone(&stop_event), Arc::clone(&conn_status));

        let image_bytes = include_bytes!("./logo.png");
        let logo = RetainedImage::from_image_bytes("LiberTEM logo", image_bytes).unwrap();

        Self {
            ring_params: Default::default(),
            data_source,
            acquisitions: HashMap::new(),
            acquisition_history: Vec::new(),
            stop_event,
            stats: AggregateStats::new(),
            previous_stats: None,
            bg_thread: Some(join_handle),
            control_channel,
            conn_status,
            logo,
        }
    }

    pub fn connect(
        stop_event: Arc<AtomicBool>,
        status: Arc<Mutex<ConnectionStatus>>,
    ) -> (JoinHandle<()>, Receiver<AcqMessage>, Sender<ControlMessage>) {
        // start background thread
        let (s, data_source) = unbounded::<AcqMessage>();

        let (control_channel, rc) = unbounded::<ControlMessage>();

        let join_handle = std::thread::Builder::new()
            .name("background".to_string())
            .spawn(move || {
                background_thread(rc, s, stop_event, status);
            })
            .unwrap();

        (join_handle, data_source, control_channel)
    }
}

impl TemplateApp {
    /// Applies one or more messages.

    fn apply_acq_message(&mut self, msg: AcqMessage) -> Option<ProcessingStats> {
        match msg {
            AcqMessage::AcquisitionStarted(_started) => {}
            AcqMessage::AcquisitionEnded(_ended) => {
                // meh
                //self.acquisitions.remove(&ended.id);
                trace!(
                    "before: {} {}",
                    self.acquisition_history.len(),
                    self.acquisitions.len()
                );
                let len = self.acquisition_history.len();
                let max_size = 2;
                if len > max_size {
                    self.acquisition_history
                        .drain(..len - max_size)
                        .for_each(|k| {
                            trace!("removing {k:?}");
                            self.acquisitions.remove(&k);
                        });
                }
                trace!(
                    "after: {} {}",
                    self.acquisition_history.len(),
                    self.acquisitions.len()
                );
            }
            AcqMessage::Stats(stats) => {
                return Some(stats);
            }
            AcqMessage::UpdatedData {
                id,
                img,
                bbox,
                udf_name,
                channel_name,
                data,
            } => {
                trace!("inserting id {id}");
                let key = ResultId {
                    acquisition: id,
                    udf_name,
                    channel_name,
                };
                if self
                    .acquisitions
                    .insert(key.clone(), (ImageOrTexture::Image(img), bbox, data))
                    .is_none()
                {
                    self.acquisition_history.push(key);
                }
            }
        }

        None
    }

    fn apply_acq_messages(&mut self) {
        let mut new_stats: Vec<ProcessingStats> = Vec::new();

        while !self.data_source.is_empty() {
            if let Ok(msg) = self.data_source.try_recv() {
                trace!("{msg:?}");
                if let Some(stats) = self.apply_acq_message(msg) {
                    new_stats.push(stats);
                }
            }
        }

        for stats in new_stats {
            self.update_stats(stats);
        }
    }

    fn update_stats(&mut self, stats: ProcessingStats) {
        self.stats.aggregate(&stats);

        let delta_t = 1.0f32;
        match &self.previous_stats {
            None => {
                self.previous_stats = Some((self.stats.clone(), self.stats.clone()));
            }
            Some((_older, newer)) => {
                if newer.ts.elapsed() > Duration::from_secs_f32(delta_t) {
                    // drop older and replace newer:
                    self.previous_stats =
                        Some((newer.clone(), self.stats.clone().with_updated_ts()));
                }
            }
        }
    }

    /// Make a list of unique (udf, channel) pairs for ordering the plots
    fn list_channels(&self) -> Vec<(String, String)> {
        let pairs = self
            .acquisition_history
            .iter()
            .map(|result_id| (result_id.udf_name.clone(), result_id.channel_name.clone()))
            .collect::<HashSet<_>>();
        let mut pairs = Vec::from_iter(pairs.into_iter());
        pairs.sort();

        pairs
    }

    fn load_textures(&mut self, ui: &mut Ui) {
        // maybe load textures (if they aren't already):
        let aq = std::mem::take(&mut self.acquisitions);
        self.acquisitions = aq
            .into_iter()
            .map(|(id, (img_or_texture, bbox, data))| {
                let tex = match img_or_texture {
                    ImageOrTexture::Image(img) => {
                        let tex_options: TextureOptions = TextureOptions {
                            magnification: egui::TextureFilter::Nearest,
                            minification: egui::TextureFilter::Linear,
                        };
                        ui.ctx().load_texture(format!("{id:?}"), img, tex_options)
                    }
                    ImageOrTexture::Texture(t) => t,
                };
                (id, (ImageOrTexture::Texture(tex), bbox, data))
            })
            .collect();
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.apply_acq_messages();
        ctx.request_repaint(); // "continuous mode"

        let Self { ring_params, .. } = self;

        // Examples of how to create different panels and windows.
        // Pick whichever suits you.
        // Tip: a good default choice is to just keep the `CentralPanel`.
        // For inspiration and more examples, go to https://emilk.github.io/egui

        #[cfg(not(target_arch = "wasm32"))] // no File->Quit on web pages!
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            // The top panel is often a good place for a menu bar:
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Quit").clicked() {
                        frame.close();
                    }
                });
            });
        });

        let send_ring_params = |ring_params: &RingParams, channel: &Sender<ControlMessage>| {
            channel
                .send(ControlMessage::UpdateParams {
                    params: UpdateParams {
                        parameters: UpdateParamsInner {
                            cx: ring_params.cx,
                            cy: ring_params.cy,
                            ri: ring_params.ri,
                            ro: ring_params.ro,
                        },
                    },
                })
                .unwrap();
        };

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Parameters");

            let ri = egui::Slider::new(&mut ring_params.ri, 0.0..=516.0 / 2.0).text("ri");
            if ui.add(ri).changed() {
                send_ring_params(ring_params, &self.control_channel);
            }

            let ro = egui::Slider::new(&mut ring_params.ro, 0.0..=516.0 / 2.0).text("ro");
            if ui.add(ro).changed() {
                send_ring_params(ring_params, &self.control_channel);
            }

            let cx = egui::Slider::new(&mut ring_params.cx, 0.0..=516.0).text("cx");
            if ui.add(cx).changed() {
                send_ring_params(ring_params, &self.control_channel);
            }

            let cy = egui::Slider::new(&mut ring_params.cy, 0.0..=516.0).text("cy");
            if ui.add(cy).changed() {
                send_ring_params(ring_params, &self.control_channel);
            }

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                self.logo.show_max_size(ui, vec2(250.0, 200.0));

                ui.label(format!(
                    "Connection: {:?}",
                    self.conn_status.lock().unwrap()
                ));
                ui.label(format!("Pending messages: {:?}", self.stats.num_pending));
                ui.label(format!("Finished Acquisitions: {:?}", self.stats.finished));

                if let Some((older, newer)) = &self.previous_stats {
                    let delta_f = (newer.finished - older.finished) as f32;
                    let delta_t = (newer.ts - older.ts).as_secs_f32();
                    let rate = delta_f / delta_t;
                    ui.label(format!("Update rate: {:.3}/s", rate));
                }

                // ui.horizontal(|ui| {
                //     ui.spacing_mut().item_spacing.x = 0.0;
                //     ui.label("powered by ");
                //     ui.hyperlink_to("egui", "https://github.com/emilk/egui");
                //     ui.label(" and ");
                //     ui.hyperlink_to(
                //         "eframe",
                //         "https://github.com/emilk/egui/tree/master/crates/eframe",
                //     );
                //     ui.label(".");
                // });
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let legend = Legend::default()
                .position(Corner::RightBottom)
                .background_alpha(0.75);

            let plot = Plot::new("items_demo")
                .legend(legend)
                .show_x(true)
                .show_y(true)
                .x_grid_spacer(Box::new(|_grid_input| Vec::new()))
                .y_grid_spacer(Box::new(|_grid_input| Vec::new()))
                .data_aspect(1.0);

            self.load_textures(ui);
            let channel_list = self.list_channels();

            plot.show(ui, |plot_ui| {
                self.acquisition_history.iter().for_each(|id| {
                    let idx = channel_list
                        .iter()
                        .position(|(udf, channel)| {
                            id.udf_name == *udf && id.channel_name == *channel
                        })
                        .unwrap();
                    if let Some(item) = self.acquisitions.get(id) {
                        let (img_or_texture, bbox, _data) = item;
                        if let ImageOrTexture::Texture(texture_id) = img_or_texture {
                            let pos =
                                PlotPoint::new(idx as f32 + texture_id.aspect_ratio() / 2.0, 0.5);
                            let image = PlotImage::new(
                                texture_id,
                                pos,
                                vec2(texture_id.aspect_ratio(), 1.0),
                            );
                            let img = image; // .name(format!("Acquisition {id:?}"));
                            plot_ui.image(img);
                            if false {
                                plot_ui.hline(
                                    HLine::new(1.0 - (bbox.ymax as f32 / texture_id.size_vec2().y))
                                        .name("Scan"),
                                );
                            }
                        }
                    }
                });

                let plot_cx = (self.ring_params.cx / 516.0 + 1.0) as f64;
                let plot_cy = 1.0 - (self.ring_params.cy / 516.0) as f64;

                let marker = Points::new(vec![[plot_cx, plot_cy]])
                    .filled(true)
                    .radius(3.0)
                    .shape(MarkerShape::Square);

                let polygon: Polygon = Polygon::new(circle_points(
                    plot_cx,
                    plot_cy,
                    self.ring_params.ri as f64 / 516.0,
                ))
                .fill_alpha(0.0);
                plot_ui.polygon(polygon);

                let polygon: Polygon = Polygon::new(circle_points(
                    plot_cx,
                    plot_cy,
                    self.ring_params.ro as f64 / 516.0,
                ))
                .fill_alpha(0.0);
                plot_ui.polygon(polygon);

                plot_ui.points(marker);
            })
            .response
        });
    }

    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, _storage: &mut dyn eframe::Storage) {
        // eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.stop_event.store(true, Relaxed);
        if let Some(_bg_thread) = self.bg_thread.take() {
            // bg_thread.join().unwrap();
            // FIXME: need to debug this
        }
    }
}
