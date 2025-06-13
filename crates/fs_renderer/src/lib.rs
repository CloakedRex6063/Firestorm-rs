use std::mem;
use std::ffi::c_void;
use crate::dx_render::RenderContextDX;
pub use crate::render_context::RenderContext;
use crate::render_structs::{RenderPass, Vertex};
use enumflags2::{BitFlags, bitflags};
use fs_handles::Handle;
use fs_window::Window;
use std::mem::ManuallyDrop;

mod dx_render;
mod geom_pass;
mod render_context;
mod render_structs;
mod shaders;

#[derive(Debug)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
}

pub struct Viewport {
    dimensions: [f32; 2],
    offset: [f32; 2],
    depth_range: [f32; 2],
}

pub struct Scissor {
    min: [i32; 2],
    max: [i32; 2],
}

#[Handle]
pub struct TextureHandle(pub u32);
#[Handle]
pub struct BufferHandle(pub u32);
#[Handle]
pub struct ShaderHandle(pub u32);
#[Handle]
pub struct CommandHandle(pub u32);
#[Handle]
pub struct ResourceHandle(pub u32);

#[derive(Clone, Copy, PartialEq)]
pub enum Format {
    RGBA8UNORM,
    RGBA16,
    D32,
    Unknown,
}

#[derive(Clone, Copy, PartialEq)]
pub enum ResourceState {
    Common,
    RenderTarget,
    DepthStencil,
    ShaderResource,
    TransferSrc,
    TransferDst,
}

pub struct AdapterInfo {
    pub name: String,
    pub vram: u64,
    pub sys_mem: u64,
}

#[bitflags]
#[repr(u8)]
#[derive(Clone, Copy, PartialEq)]
pub enum TextureFlags {
    RenderTarget = 1 << 0,
    DepthStencil = 1 << 1,
    ShaderResource = 1 << 2,
}

pub struct TextureCreateInfo {
    pub width: u32,
    pub height: u32,
    pub array_layers: u16,
    pub mip_levels: u16,
    pub format: Format,
    pub texture_flags: BitFlags<TextureFlags>,
    pub data: Vec<u8>,
}

pub struct BufferCreateInfo {
    pub stride: u32,
    pub elements: u32,
}

pub struct ComputeShaderCreateInfo {
    shader_code: Vec<u8>,
}

pub struct TradShaderCode {
    vertex_code: Option<Vec<u8>>,
    geom_code: Option<Vec<u8>>,
    hull_code: Option<Vec<u8>>,
    domain_code: Option<Vec<u8>>,
}

pub struct MeshShaderCode {
    task_code: Option<Vec<u8>>,
    mesh_code: Option<Vec<u8>>,
}

pub union GraphicsShaderCode {
    trad_shader_code: ManuallyDrop<TradShaderCode>,
    mesh_shader_code: ManuallyDrop<MeshShaderCode>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    TriangleList,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FillMode {
    Solid,
    Wireframe,
}

#[derive(Clone, Copy, PartialEq)]
pub enum CullMode {
    None,
    Front,
    Back,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FrontFace {
    Clockwise,
    CounterClockwise,
}

pub struct GraphicsShaderCreateInfo {
    graphics_shader_code: GraphicsShaderCode,
    pixel_code: Option<Vec<u8>>,
    rtv_formats: Vec<Format>,
    dsv_format: Format,
    primitive_topology: PrimitiveTopology,
    fill_mode: FillMode,
    cull_mode: CullMode,
    front_face: FrontFace,
}

pub enum RenderBackendType {
    DX12,
}

#[derive(Debug)]
pub enum Error {
    CreateFactory,
    GetAdapter,
    GetBackBuffer,
    #[cfg(debug_assertions)]
    CreateDebugContext,
    #[cfg(debug_assertions)]
    EnableDebugLayer,
    CreateSwapchain,
    CreateCommandQueue,
    CreateCommand,
    CreateFence,
    CreateBuffer,
    CreateTexture,
    CreateShader,
    CreatePipeline,
    CreateRootSignature,
    CreateDescriptorHeap,
    CreateRenderPass,
    MapBuffer
}

#[macro_export] macro_rules! get_ptr_and_size {
    ($val:expr) => {{
        let ref_val = &$val;
        (ref_val.as_ptr() as *const c_void, (ref_val.len() * mem::size_of_val(&ref_val[0])) as usize)
    }};
}

pub struct Renderer {
    render_context: Box<dyn RenderContext>,
    render_passes: Vec<RenderPass>,
    render_target: TextureHandle,
    depth_stencil: TextureHandle,
    vsync: bool,
}

impl Renderer {
    pub fn new(backend: RenderBackendType, window: &Window) -> Result<Renderer, Error> {
        let mut render_context = match backend {
            RenderBackendType::DX12 => RenderContextDX::new(window)?,
        };

        let mut tex_create_info = TextureCreateInfo {
            width: window.get_size().x as u32,
            height: window.get_size().y as u32,
            array_layers: 1,
            mip_levels: 1,
            format: Format::RGBA8UNORM,
            texture_flags: TextureFlags::RenderTarget.into(),
            data: vec![],
        };
        let render_target = render_context.create_texture(&tex_create_info)?;
        tex_create_info.format = Format::D32;
        tex_create_info.texture_flags = TextureFlags::DepthStencil.into();
        let depth_stencil = render_context.create_texture(&tex_create_info)?;

        let mut renderer = Self {
            render_context,
            render_passes: vec![],
            render_target,
            depth_stencil,
            vsync: false,
        };
        renderer.add_default_render_passes()?;
        Ok(renderer)
    }

    pub fn get_render_context(&mut self) -> &mut dyn RenderContext {
        self.render_context.as_mut()
    }

    pub fn get_render_context_ref(&self) -> &dyn RenderContext {
        self.render_context.as_ref()
    }

    pub fn get_render_passes(&self) -> &Vec<RenderPass> {
        self.render_passes.as_ref()
    }

    pub fn get_render_passes_mut(&mut self) -> &mut Vec<RenderPass> {
        self.render_passes.as_mut()
    }

    pub fn add_render_pass(&mut self, render_pass: RenderPass) {
        self.render_passes.push(render_pass);
    }

    fn add_default_render_passes(&mut self) -> Result<(), Error> {
        let vertex_buffer_info = BufferCreateInfo {
            stride: size_of::<Vertex>() as u32,
            elements: 3,
        };
        let vertex_buffer = self.render_context.create_buffer(&vertex_buffer_info)?;

        let vertex_data: [Vertex; 3] = [
            Vertex {
                position: [-0.5f32, -0.5f32, 0.0f32],
                ..Default::default()
            },
            Vertex {
                position: [0.5f32, -0.5f32, 0.0f32],
                ..Default::default()
            },
            Vertex {
                position: [0f32, 0.5f32, 0.0f32],
                ..Default::default()
            },
        ];
        let (ptr, len) = get_ptr_and_size!(vertex_data);
        self.render_context.write_buffer(vertex_buffer, ptr, len)?;
        geom_pass::add_geom_pass(self, vertex_buffer)?;
        Ok(())
    }

    pub fn toggle_vsync(&mut self, vsync: bool) {
        self.vsync = vsync;
    }

    pub fn render(self: &mut Renderer) {
        let context = self.render_context.as_mut();
        let command = context.get_current_frame().command_id;

        context.begin_command(command);

        self.render_passes.iter().for_each(|pass| {
            (pass.pass)(context, pass);
        });

        context.copy_to_swapchain(command, self.render_target);

        context.end_command(command);

        context.submit_command(command, QueueType::Graphics);
        context.present(self.vsync);
    }
}


