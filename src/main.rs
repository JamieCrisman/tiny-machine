use std::{
    collections::HashMap,
    env,
    time::{Duration, Instant},
};

use game_loop::{game_loop, Time, TimeTrait};
use pixels::{Error, Pixels, SurfaceTexture};
use vm::VM;
use winit::{
    dpi::LogicalSize,
    event::VirtualKeyCode,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use winit_input_helper::WinitInputHelper;

mod gui;
use crate::gui::Gui;

mod compiler;
mod vm;

// GB inspired clock speed for the moment
const CLOCK_SPEED: u64 = 4194304;
const FRAME_RATE: u64 = 60;
const CYCLES_PER_FRAME: u64 = CLOCK_SPEED / FRAME_RATE;
const TIME_STEP: Duration = Duration::from_nanos(1_000_000_000 / FRAME_RATE as u64);

// const WIDTH: i32 = 320;
// const HEIGHT: i32 = 240;
const WIDTH: i32 = 160;
const HEIGHT: i32 = 144;

#[derive(Clone)]
struct Stats {
    process_time: Duration,
    process_time_list: Vec<f32>,
    cycles: u64,
    stack_size: usize,
    // stack_length: usize,
    globals_size: usize,
    // globals_length: usize,
    ip: i64,
    sp: usize,
    op_times: HashMap<String, Duration>,
}

struct Machine {
    /// Software renderer.
    pixels: Pixels,
    // /// Invaders world.
    // world: World,
    /// Player controls for world updates.
    // controls: Controls,
    /// Event manager.
    input: WinitInputHelper,
    /// GamePad manager.
    // gilrs: Gilrs,
    /// GamePad ID for the player.
    // gamepad: Option<GamepadId>,
    /// Game pause state.
    // paused: bool,
    vm: VM,
    gui: gui::Gui,
}

impl Machine {
    fn new(mut pixels: Pixels, bc: compiler::Bytecode, gui: gui::Gui, debug: bool) -> Self {
        let vm = VM::flash(bc, pixels.get_frame().len() / 4, debug);
        Self {
            pixels,
            // controls: Controls::default(),
            input: WinitInputHelper::new(),
            // gilrs: Gilrs::new().unwrap(), // XXX: Don't unwrap.
            // gamepad: None,
            vm,
            gui,
        }
    }

    fn update_controls(&mut self) {
        // Pump the gilrs event loop and find an active gamepad
        // while let Some(gilrs::Event { id, event, .. }) = self.gilrs.next_event() {
        //     let pad = self.gilrs.gamepad(id);
        //     if self.gamepad.is_none() {
        //         debug!("Gamepad with id {} is connected: {}", id, pad.name());
        //         self.gamepad = Some(id);
        //     } else if event == gilrs::ev::EventType::Disconnected {
        //         debug!("Gamepad with id {} is disconnected: {}", id, pad.name());
        //         self.gamepad = None;
        //     }
        // }

        // self.controls = {
        //     // Keyboard controls
        //     let mut left = self.input.key_held(VirtualKeyCode::Left);
        //     let mut right = self.input.key_held(VirtualKeyCode::Right);
        //     let mut fire = self.input.key_pressed(VirtualKeyCode::Space);
        //     let mut pause = self.input.key_pressed(VirtualKeyCode::Pause)
        //         | self.input.key_pressed(VirtualKeyCode::P);

        //     // GamePad controls
        //     if let Some(id) = self.gamepad {
        //         let gamepad = self.gilrs.gamepad(id);

        //         left |= gamepad.is_pressed(Button::DPadLeft);
        //         right |= gamepad.is_pressed(Button::DPadRight);
        //         fire |= gamepad.button_data(Button::South).map_or(false, |button| {
        //             button.is_pressed() && button.counter() == self.gilrs.counter()
        //         });
        //         pause |= gamepad.button_data(Button::Start).map_or(false, |button| {
        //             button.is_pressed() && button.counter() == self.gilrs.counter()
        //         });
        //     }
        //     self.gilrs.inc();

        //     if pause {
        //         self.paused = !self.paused;
        //     }

        //     let direction = if left {
        //         Direction::Left
        //     } else if right {
        //         Direction::Right
        //     } else {
        //         Direction::Still
        //     };

        //     Controls { direction, fire }
        // };
    }
}

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    // Enable debug mode with `DEBUG=true` environment variable
    let debug = env::var("DEBUG")
        .unwrap_or_else(|_| "false".to_string())
        .parse()
        .unwrap_or(false);

    let window = {
        let size = LogicalSize::new(WIDTH as f64, HEIGHT as f64);
        let scaled_size = LogicalSize::new(WIDTH as f64 * 4.0, HEIGHT as f64 * 4.0);
        WindowBuilder::new()
            .with_title("VM")
            .with_inner_size(scaled_size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture)?
    };

    let bc = compiler::Bytecode {
        instructions: compiler::Instructions {
            data: vec![
                // initial values
                6, //
                2, 0, 0, //
                2, 0, 1, //
                // store initial values
                10, 0, 0, // 0
                10, 0, 1, // 255
                10, 0, 2, // 0
                10, 0, 3, // 0
                // put current values on stack
                9, 0, 3, // 0
                9, 0, 2, // 0
                9, 0, 1, // 255
                // render pixel
                8, // 255, 0, 0
                // clear accumulator
                3, //
                // load pixel color to accumulator
                9, 0, 1, //
                9, 0, 0, //
                7, //
                // increment accumulator and push to stack
                4, //
                6, //
                // save pixel color
                10, 0, 0, //
                10, 0, 1, //
                // load pixel location to accumulator
                9, 0, 3, //
                9, 0, 2, //
                7, //
                // increment accumulator
                4, //
                6, //
                // save new pixel location
                10, 0, 2, //
                10, 0, 3, //
                1, 0xFF, 0xD6, // jump to putting current values on stack
            ],
        },
        constants: vec![255, 0],
    };

    let mut gui = Gui::new(&window, &pixels);

    let machine = Machine::new(pixels, bc, gui, debug);

    game_loop(
        event_loop,
        window,
        machine,
        FRAME_RATE as u32,
        TIME_STEP.as_secs_f64(),
        move |g| {
            // Update the world
            // if !g.game.paused {
            //     g.game.world.update(&g.game.controls);
            // }
            let mut emulated_cycles: u64 = 0;
            let start = Instant::now();
            while emulated_cycles <= CYCLES_PER_FRAME
                && start.elapsed().as_secs_f32() < (TIME_STEP.as_secs_f32() * 0.75)
            {
                match g.game.vm.execute() {
                    Ok(step) => {
                        emulated_cycles += step as u64;
                    }
                    Err(err) => panic!("error on execution: {:?}", err),
                }
            }
            // g.game.stats.process_time = start.elapsed();
            g.game.gui.set_stats(
                start.elapsed(),
                emulated_cycles,
                g.game.vm.stack_size(),
                g.game.vm.globals_size(),
                g.game.vm.sp(),
                g.game.vm.ip(),
                g.game.vm.operation_times(),
            );
        },
        move |g| {
            let render_time = Instant::now();
            g.game.gui.prepare(&g.window).expect("gui.prepare() failed");

            // Drawing
            g.game.vm.update_screen(g.game.pixels.get_frame());
            // Render everything together
            let render_result = g
                .game
                .pixels
                .render_with(|encoder, render_target, context| {
                    // Render the world texture
                    context.scaling_renderer.render(encoder, render_target);

                    // Render Dear ImGui
                    g.game
                        .gui
                        .render(&g.window, encoder, render_target, context)?;

                    Ok(())
                });

            if render_result.is_err() {
                panic!(
                    "got unexpected render error {}",
                    render_result.err().unwrap()
                );
            }
            // if let Err(e) = g.game.pixels.render() {
            //     // error!("pixels.render() failed: {}", e);
            //     panic!("pixels render failed: {}", e);
            //     // g.exit();
            // }

            // Sleep the main thread to limit drawing to the fixed time step.
            // See: https://github.com/parasyte/pixels/issues/174
            let dt = TIME_STEP.as_secs_f64() - Time::now().sub(&g.current_instant());
            // let dt = TIME_STEP.as_secs_f64() - render_time.elapsed().as_secs_f64();
            if dt > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(dt));
            }
        },
        |g, event| {
            // Let winit_input_helper collect events to build its state.
            g.game.gui.handle_event(&g.window, &event);

            if g.game.input.update(event) {
                // Update controls
                g.game.update_controls();

                // Close events
                if g.game.input.key_pressed(VirtualKeyCode::Escape) || g.game.input.quit() {
                    g.exit();
                    return;
                }

                if g.game.input.key_released(VirtualKeyCode::Grave) {
                    g.game.gui.toggle();
                }

                // Resize the window
                if let Some(size) = g.game.input.window_resized() {
                    g.game.pixels.resize_surface(size.width, size.height);
                }
            }
        },
    );
}
