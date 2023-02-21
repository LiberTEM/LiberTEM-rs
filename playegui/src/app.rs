use std::{collections::{HashMap, hash_map::Entry}, time::Duration};

use crossbeam::channel::{Receiver, unbounded};
use egui::{
    plot::{Corner, HLine, Legend, Plot, PlotImage, PlotPoint, VLine, Polygon},
    vec2, TextureHandle, ColorImage, TextureOptions,
};
use log::{info, trace};
use ndarray::Array2;

use crate::{messages::{AcqMessage, BBox}, background::background_thread};

/// We derive Deserialize/Serialize so we can persist app state on shutdown.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(default)] // if we add new fields, give them default values when deserializing old state
pub struct TemplateApp {
    // Example stuff:
    label: String,

    // this how you opt-out of serialization of a member
    #[serde(skip)]
    value: f32,

    #[serde(skip)]
    data_source: Option<Receiver<AcqMessage>>,

    #[serde(skip)]
    acquisitions: HashMap<String, (ColorImage, Option<TextureHandle>, BBox)>,

    #[serde(skip)]
    acquisition_history: Vec<String>,
}

impl Default for TemplateApp {
    fn default() -> Self {
        Self {
            label: Default::default(),
            value: Default::default(),
            data_source: Default::default(),
            acquisitions: HashMap::new(),
            acquisition_history: Vec::new(),
        }
    }
}

impl TemplateApp {
    /// Called once before the first frame.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // This is also where you can customize the look and feel of egui using
        // `cc.egui_ctx.set_visuals` and `cc.egui_ctx.set_fonts`.

        // Load previous app state (if any).
        // Note that you must enable the `persistence` feature for this to work.
        if let Some(storage) = cc.storage {
            let mut loaded: TemplateApp = eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default();
            loaded.data_source = Some(Self::connect());
            return loaded;
        }

        Self {
            // Example stuff:
            label: "Hello World!".to_owned(),
            value: 2.7,
            data_source: Some(Self::connect()),
            acquisitions: HashMap::new(),
            acquisition_history: Vec::new(),
        }
    }

    pub fn connect() -> Receiver<AcqMessage> {
        // start background thread
        let (s, data_source) = unbounded::<AcqMessage>();

        std::thread::spawn(move || {
            background_thread(s);
        });

        data_source
    }
}

impl TemplateApp {
    /// Applies at most one pending message. Call each frame to balance work a bit.
    fn apply_pending_messages(&mut self) {
        if let Some(receiver) = &self.data_source {
            if let Ok(msg) = receiver.try_recv() {
                trace!("{msg:?}");
                match msg {
                    AcqMessage::AcquisitionStarted(started) => {
                    }
                    AcqMessage::AcquisitionEnded(ended) => {
                        // meh
                        //self.acquisitions.remove(&ended.id);
                        trace!("before: {} {}", self.acquisition_history.len(), self.acquisitions.len());
                        let len = self.acquisition_history.len();
                        let max_size = 2;
                        if len > max_size {
                            self.acquisition_history.drain(..len - max_size).for_each(|k| {
                                trace!("removing {k}");
                                self.acquisitions.remove(&k);
                            });
                        }
                        trace!("after: {} {}", self.acquisition_history.len(), self.acquisitions.len());
                    }
                    AcqMessage::AcquisitionResult(..) => {
                        // mmmmeh? bg thread will create texture for us in different msg...
                    }
                    AcqMessage::UpdatedData(id, data, bbox) => {
                        trace!("inserting id {id}");
                        if self.acquisitions.insert(id.clone(), (data, None, bbox)).is_none() {
                            self.acquisition_history.push(id);
                        }
                    }
                }
            }
        }
    }
}

impl eframe::App for TemplateApp {
    /// Called each time the UI needs repainting, which may be many times per second.
    /// Put your widgets into a `SidePanel`, `TopPanel`, `CentralPanel`, `Window` or `Area`.
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.apply_pending_messages();
        ctx.request_repaint();  // "continuous mode"

        let Self { label, value, .. } = self;

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

        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.heading("Side Panel");

            ui.horizontal(|ui| {
                ui.label("Write something: ");
                ui.text_edit_singleline(label);
            });

            ui.add(egui::Slider::new(value, 0.0..=10.0).text("value"));
            if ui.button("Increment").clicked() {
                *value += 1.0;
            }

            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    ui.label("powered by ");
                    ui.hyperlink_to("egui", "https://github.com/emilk/egui");
                    ui.label(" and ");
                    ui.hyperlink_to(
                        "eframe",
                        "https://github.com/emilk/egui/tree/master/crates/eframe",
                    );
                    ui.label(".");
                });
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
                .x_grid_spacer(Box::new(|x| Vec::new()))
                .y_grid_spacer(Box::new(|x| Vec::new()))
                .data_aspect(1.0);

            // maybe load textures
            self.acquisitions.iter_mut().for_each(|(id, (img, texture, bbox))| {
                // FIXME: remove clone? does it matter?
                let tex_options: TextureOptions = TextureOptions { magnification: egui::TextureFilter::Nearest, minification: egui::TextureFilter::Linear };
                let handle = ui.ctx().load_texture(id, img.clone(), tex_options);
                *texture = Some(handle);
            });

            plot.show(ui, |plot_ui| {
                // plot_ui.hline(HLine::new(9.0).name("Lines horizontal"));
                // plot_ui.hline(HLine::new(-9.0).name("Lines horizontal"));
                // plot_ui.vline(VLine::new(9.0).name("Lines vertical"));
                // plot_ui.vline(VLine::new(-9.0).name("Lines vertical"));
                //plot_ui.image(image.name("Image"));

                self.acquisition_history.iter().rev().take(2).rev().for_each(|id| {
                    if let Some(item) = self.acquisitions.get(id) {
                        let (_, texture, bbox) = item;
                        let texture_id = texture.as_ref().unwrap();
                        let pos = PlotPoint::new(texture_id.aspect_ratio() / 2.0, 0.5);
                        let image = PlotImage::new(
                            texture_id,
                            pos,
                            vec2(texture_id.aspect_ratio(), 1.0),
                        );
                        let img = image.name(format!("Acquisition {id}"));
                        plot_ui.image(img);

                        // println!("{:?}", bbox);
                        plot_ui.hline(HLine::new(1.0 - (bbox.ymax as f32 / texture_id.size_vec2().y)).name("Scan"));
                    }
                });

                // if let Some(last) = self.acquisitions.iter().last() {
                //     let (id, (_, texture)) = last;
                //     let texture_id = texture.as_ref().unwrap();
                //     let pos = PlotPoint::new(texture_id.aspect_ratio() / 2.0, 0.5);
                //     let image = PlotImage::new(
                //         texture_id,
                //         pos,
                //         vec2(texture_id.aspect_ratio(), 1.0),
                //     );
                //     let img = image.name(format!("Acquisition {id}"));
                //     plot_ui.image(img);
                // }

                // self.acquisitions.iter().for_each(|(id, (_, texture))| {
                //     let texture_id = texture.as_ref().unwrap();
                //     let image = PlotImage::new(
                //         texture_id,
                //         PlotPoint::new((5.0 * texture_id.aspect_ratio()) / 2.0, 2.5),
                //         vec2(texture_id.aspect_ratio(), 1.0),
                //     );
                //     let img = image.name(id);
                //     plot_ui.image(img);
                // });
            })
            .response
        });

        if false {
            egui::Window::new("Window").show(ctx, |ui| {
                ui.label("Windows can be moved by dragging them.");
                ui.label("They are automatically sized based on contents.");
                ui.label("You can turn on resizing and scrolling if you like.");
                ui.label("You would normally choose either panels OR windows.");
            });
        }
    }

    /// Called by the frame work to save state before shutdown.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }
}
