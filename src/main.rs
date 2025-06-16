use fs_engine::Engine;

fn main() {
    let mut engine = Engine::new().expect("Failed to create engine");
    engine.resources.load_model(&mut engine.renderer, "assets/DamagedHelmet.glb").expect("Failed to load model");
    engine.run();
}

