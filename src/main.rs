use env_logger::{Builder, Env};
use fs_renderer::*;
use fs_window as window;
use log::error;
use std::time::Instant;

fn main() {
    #[cfg(debug_assertions)]
    let default_log_level = "debug";
    #[cfg(not(debug_assertions))]
    let default_log_level = "info";
    Builder::from_env(Env::default().default_filter_or(default_log_level)).init();

    let mut window = window::Window::new("Firestorm");
    window.register_window();
    let renderer_result = Renderer::new(RenderBackendType::DX12, &window);
    if renderer_result.is_err() {
        error!("Failed to create renderer: {:?}", renderer_result.err());
        return;
    }
    let mut renderer = renderer_result.unwrap();
    renderer.toggle_vsync(false);

    let mut prev = Instant::now();

    while window.is_open() {
        window.poll_events();
        renderer.render();
        let now = Instant::now();
        let elapsed = now - prev;
        prev = now;
        window.set_title(&format!("FPS: {}", 1.0 / elapsed.as_secs_f32()));
    }
}
