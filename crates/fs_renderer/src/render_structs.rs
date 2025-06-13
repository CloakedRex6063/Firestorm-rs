use crate::{BufferHandle, Format, RenderContext, Renderer, ShaderHandle, TextureHandle};

pub enum RenderPassError {
    None,
    NoOutput,
    NoShader,
    NoPass,
}

#[derive(Clone)]
pub enum RenderPassInput {
    Buffer(BufferHandle),
    Texture(TextureHandle),
}

pub struct RenderPass {
    pub pass: fn(&mut dyn RenderContext, &RenderPass),
    pub shader: ShaderHandle,
    pub inputs: Vec<RenderPassInput>,
    pub outputs: Vec<TextureHandle>,
    pub depth: Option<TextureHandle>,
}

pub struct RenderPassBuilder<'a> {
    renderer: &'a Renderer,
    pass: Option<fn(&mut dyn RenderContext, &RenderPass)>,
    shader: Option<ShaderHandle>,
    inputs: Vec<RenderPassInput>,
    outputs: Vec<TextureHandle>,
    depth: Option<TextureHandle>,
}

impl<'a> RenderPassBuilder<'a> {
    pub fn new(renderer: &'a Renderer) -> Self {
        RenderPassBuilder {
            renderer,
            pass: None,
            shader: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            depth: None,
        }
    }

    pub fn set_shader(&mut self, shader: ShaderHandle) -> &mut Self {
        self.shader = Some(shader);
        self
    }

    pub fn add_input(&mut self, input: RenderPassInput) -> &mut Self {
        self.inputs.push(input);
        self
    }

    pub fn add_output(&mut self, output: TextureHandle) -> &mut Self {
        if self.renderer.render_context.get_texture_format(output) == Format::D32 {
            self.depth = Some(output);
        } else {
            self.outputs.push(output);
        }
        self
    }

    pub fn add_pass(&mut self, pass: fn(&mut dyn RenderContext, &RenderPass)) -> &mut Self {
        self.pass = Some(pass);
        self
    }

    pub fn build(&self) -> Result<RenderPass, RenderPassError> {
        if self.pass.is_none() {
            Err(RenderPassError::NoPass)?
        };
        if self.outputs.is_empty() {
            Err(RenderPassError::NoOutput)?
        }
        if self.shader.is_none() {
            Err(RenderPassError::None)?
        }
        Ok(RenderPass {
            pass: self.pass.unwrap(),
            shader: self.shader.unwrap(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            depth: self.depth,
        })
    }
}

#[repr(C)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_x: f32,
    pub normal: [f32; 3],
    pub uv_y: f32,
    pub tangent: [f32; 4],
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            position: [0.0, 0.0, 0.0],
            uv_x: 0.0,
            normal: [0.0, 0.0, 0.0],
            uv_y: 0.0,
            tangent: [0.0, 0.0, 0.0, 0.0],
        }
    }
}
