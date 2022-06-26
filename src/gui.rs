use pixels::{wgpu, PixelsContext};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

use crate::Stats;

pub(crate) struct Gui {
    imgui: imgui::Context,
    platform: imgui_winit_support::WinitPlatform,
    renderer: imgui_wgpu::Renderer,
    last_frame: Instant,
    last_cursor: Option<imgui::MouseCursor>,
    about_open: bool,
    metrics_open: bool,
    stats_open: bool,
    visible: bool,
    stats: Stats,
}

impl Gui {
    /// Create Dear ImGui.
    pub(crate) fn new(window: &winit::window::Window, pixels: &pixels::Pixels) -> Self {
        // Create Dear ImGui context
        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);

        // Initialize winit platform support
        let mut platform = imgui_winit_support::WinitPlatform::init(&mut imgui);
        platform.attach_window(
            imgui.io_mut(),
            window,
            imgui_winit_support::HiDpiMode::Default,
        );

        // Configure Dear ImGui fonts
        let hidpi_factor = window.scale_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;
        imgui
            .fonts()
            .add_font(&[imgui::FontSource::DefaultFontData {
                config: Some(imgui::FontConfig {
                    oversample_h: 1,
                    pixel_snap_h: true,
                    size_pixels: font_size,
                    ..Default::default()
                }),
            }]);

        // Create Dear ImGui WGPU renderer
        let device = pixels.device();
        let queue = pixels.queue();
        let config = imgui_wgpu::RendererConfig {
            texture_format: pixels.render_texture_format(),
            ..Default::default()
        };
        let renderer = imgui_wgpu::Renderer::new(&mut imgui, device, queue, config);

        // Return GUI context
        Self {
            imgui,
            platform,
            renderer,
            last_frame: Instant::now(),
            last_cursor: None,
            about_open: false,
            metrics_open: false,
            stats_open: true,
            visible: false,
            stats: Stats {
                process_time: Duration::new(0, 0),
                process_time_list: vec![],
                cycles: 0,
                globals_size: 0,
                ip: 0,
                sp: 0,
                stack_size: 0,
                halted: false,
                op_times: HashMap::new(),
            },
        }
    }

    /// Prepare Dear ImGui.
    pub(crate) fn prepare(
        &mut self,
        window: &winit::window::Window,
    ) -> Result<(), winit::error::ExternalError> {
        // Prepare Dear ImGui
        let now = Instant::now();
        self.imgui.io_mut().update_delta_time(now - self.last_frame);
        self.last_frame = now;
        self.platform.prepare_frame(self.imgui.io_mut(), window)
    }

    /// Render Dear ImGui.
    pub(crate) fn render(
        &mut self,
        window: &winit::window::Window,
        encoder: &mut wgpu::CommandEncoder,
        render_target: &wgpu::TextureView,
        context: &PixelsContext,
    ) -> imgui_wgpu::RendererResult<()> {
        if !self.visible {
            return Ok(());
        }
        // Start a new Dear ImGui frame and update the cursor
        let ui = self.imgui.frame();

        let mouse_cursor = ui.mouse_cursor();
        if self.last_cursor != mouse_cursor {
            self.last_cursor = mouse_cursor;
            self.platform.prepare_render(&ui, window);
        }

        // Draw windows and GUI elements here
        let mut about_open = false;
        let mut metrics_open = false;
        let mut stats_open = false;
        ui.main_menu_bar(|| {
            ui.menu("Hello", || {
                about_open = imgui::MenuItem::new("About...").build(&ui);
                metrics_open = imgui::MenuItem::new("Metrics").build(&ui);
                stats_open = imgui::MenuItem::new("Stats").build(&ui);
            });
        });
        if about_open {
            self.about_open = true;
        }
        if metrics_open {
            self.metrics_open = true;
        }
        if stats_open {
            self.stats_open = true;
            ui.open_popup("stats");
        }

        if self.about_open {
            ui.show_about_window(&mut self.about_open);
        }

        if self.metrics_open {
            ui.show_metrics_window(&mut self.metrics_open)
        }

        if self.stats_open {
            // ui.popup("stats", || {
            //     ui.text(format!("Processing Time: {:?}", self.stats.process_time));
            // });

            let from = std::cmp::max(self.stats.process_time_list.len() as i32 - 100, 0) as usize;
            let process_time: &[f32] = &self.stats.process_time_list[from..];
            ui.popup_modal("stats")
                .menu_bar(true)
                .title_bar(true)
                .opened(&mut self.stats_open)
                .always_use_window_padding(true)
                .build(&ui, || {
                    ui.plot_histogram("Processing Time", process_time).build();
                    ui.text(format!("Processing Time: {:?}", self.stats.process_time));
                    ui.text(format!("Cycles: {:?}", self.stats.cycles));
                    ui.text(format!("IP: {:?}", self.stats.ip));
                    ui.text(format!("SP: {:?}", self.stats.sp));
                    ui.text(format!("Stack Size: {:?}", self.stats.stack_size));
                    ui.text(format!("Halted: {:?}", self.stats.halted));
                    ui.text(format!("Globals Size: {:?}", self.stats.globals_size));
                    let mut op_string: String = String::new();
                    // for (op_name, dur) in self.stats.op_times.iter() {
                    //     op_string += format!("{} {:?}\n", op_name.as_str(), dur).as_str();
                    // }
                    let mut opvec: Vec<(&String, &Duration)> = self.stats.op_times.iter().collect();
                    opvec.sort_by(|a, b| b.0.cmp(a.0));
                    for (op_name, dur) in opvec.iter() {
                        op_string += format!("{} {:?}\n", op_name.as_str(), dur).as_str();
                    }
                    ui.text(format!("OpStats:\n{}\n", op_string));
                });
            // .begin_popup(&ui);
        }
        // ui.menu_with_enabled("stats", self.stats_open, || {
        // ui.text(format!("Processing Time: {:?}", self.stats.process_time))
        // });

        // Render Dear ImGui with WGPU
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("imgui"),
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: render_target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        self.renderer
            .render(ui.render(), &context.queue, &context.device, &mut rpass)
    }

    /// Handle any outstanding events.
    pub(crate) fn handle_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::Event<()>,
        // input: &mut WinitInputHelper,
    ) {
        self.platform
            .handle_event(self.imgui.io_mut(), window, event);
    }

    pub(crate) fn toggle(&mut self) {
        self.visible = !self.visible;
    }

    pub(crate) fn set_stats(
        &mut self,
        process_time: Duration,
        cycles: u64,
        stack_size: usize,
        globals_size: usize,
        sp: usize,
        ip: i64,
        op_times: HashMap<String, Duration>,
        halted: bool,
    ) {
        self.stats.process_time = process_time;
        self.stats.cycles = cycles;
        self.stats
            .process_time_list
            .push(process_time.as_secs_f32());
        self.stats.stack_size = stack_size;
        self.stats.globals_size = globals_size;
        self.stats.sp = sp;
        self.stats.ip = ip;
        self.stats.op_times.clear();
        self.stats.op_times.clone_from(&op_times);
        self.stats.halted = halted;
    }
}
