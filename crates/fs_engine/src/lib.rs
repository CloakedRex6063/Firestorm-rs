use env_logger::{Builder, Env};
use fs_renderer::*;
use fs_window as window;
use std::time::Instant;
use fs_resources::Resources;

pub struct Engine
{
    pub resources: Resources,
    pub window: window::Window,
    pub renderer: Renderer,
    pub start_time: Instant,
    pub prev_frame_time: Instant,
}

impl Engine
{
    pub fn new() -> Result<Self, Error>
    {
        #[cfg(debug_assertions)]
        let default_log_level = "debug";
        #[cfg(not(debug_assertions))]
        let default_log_level = "info";
        Builder::from_env(Env::default().default_filter_or(default_log_level)).init();

        let window = window::Window::new("Firestorm");
        let renderer = Renderer::new(RenderBackendType::DX12, &window)?;
        let resources = Resources::new();
        
        let start_time = Instant::now();
        let prev_frame_time = Instant::now();

        let engine = Self{
            resources,
            window,
            renderer,
            start_time,
            prev_frame_time,
        };
        
        Ok(engine)
    }
    
    pub fn run(&mut self)
    {
        while self.window.is_open() {
            self.window.poll_events();
            self.renderer.render(self.resources.get_models());
            let now = Instant::now();
            let elapsed = now - self.prev_frame_time;
            self.prev_frame_time = now;
            self.window.set_title(&format!("FPS: {}", 1.0 / elapsed.as_secs_f32()));
        }
    }
}