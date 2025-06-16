use fs_gltf::load_gltf;
use fs_renderer::{Model, Renderer};
use std::path::Path;
use log::info;

#[derive(Debug)]
pub enum Error {
    InvalidFile,
}

pub struct Resources {
    models: Vec<Model>,
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}

impl Resources {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
        }
    }
    
    pub fn get_models(&self) -> &[Model] {
        &self.models
    }

    pub fn load_model(&mut self, renderer: &mut Renderer, model_path: &str) -> Result<(), Error> {
        let path = Path::new(model_path);

        info!("Loading model: {}", model_path);
        
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("gltf") | Some("glb") => {
                let model = load_gltf(model_path, renderer).map_err(|_| Error::InvalidFile)?;
                self.models.push(model);
            }
            _ => Err(Error::InvalidFile)?
        };
        
        info!("Model loaded: {}", model_path);
        Ok(())
    }
}
