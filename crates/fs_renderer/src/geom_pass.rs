use std::ffi::c_void;
use crate::render_structs::{RenderPass, RenderPassBuilder, RenderPassInput};
use crate::shaders::{geom_ps, geom_vs};
use crate::{BufferHandle, CullMode, Error, FillMode, Format, FrontFace, GraphicsShaderCode, GraphicsShaderCreateInfo, PrimitiveTopology, RenderContext, Renderer, Scissor, TradShaderCode, Viewport};
use std::mem::ManuallyDrop;

fn geom_pass_render(context: &mut dyn RenderContext, render_pass: &RenderPass) {
    let command = context.get_current_frame().command_id;
    let outputs = render_pass.outputs.clone();
    context.bind_shader(command, render_pass.shader);
    for output in &outputs {
        context.clear_color(command, *output, [0f32, 0f32, 0f32, 0f32]);
    }
    let tex_size = context.get_texture_size(outputs[0]);
    let viewport = Viewport {
        dimensions: tex_size.map(|x| x as f32),
        offset: [0f32, 0f32],
        depth_range: [0f32, 1f32],
    };
    context.set_viewport(command, viewport);
    let scissor = Scissor {
        min: [0i32; 2],
        max: tex_size.map(|x| x as i32),
    };
    context.set_scissor(command, scissor);
    context.set_primitive_topology(command, PrimitiveTopology::TriangleList);

    #[repr(C)]
    struct PerDrawConstants
    {
        vertex_buffer_index: u32
    }

    let mut push_constant = PerDrawConstants
    {
        vertex_buffer_index: 0,
    };
    
    if let RenderPassInput::Buffer(buffer) = &render_pass.inputs[0] {
        push_constant = PerDrawConstants
        {
            vertex_buffer_index: context.get_gpu_address(*buffer),
        };
    }
    context.bind_render_targets(command, Some(outputs), render_pass.depth);
    context.push_constant(command, &push_constant as *const _ as *const c_void, 1);
    context.draw(command, 3, 1, 0, 0);
}

pub fn add_geom_pass(renderer: &mut Renderer, vertex_buffer: BufferHandle) -> Result<(), Error> {
    let formats = vec![Format::RGBA8UNORM];
    let create_info = GraphicsShaderCreateInfo {
        graphics_shader_code: GraphicsShaderCode {
            trad_shader_code: ManuallyDrop::new(TradShaderCode {
                vertex_code: Some(Vec::from(geom_vs)),
                geom_code: None,
                hull_code: None,
                domain_code: None,
            }),
        },
        pixel_code: Some(Vec::from(geom_ps)),
        rtv_formats: formats,
        dsv_format: Format::D32,
        primitive_topology: PrimitiveTopology::TriangleList,
        fill_mode: FillMode::Solid,
        cull_mode: CullMode::None,
        front_face: FrontFace::Clockwise,
    };
    let shader = renderer
        .render_context
        .create_graphics_shader(&create_info)?;

    let render_pass = RenderPassBuilder::new(renderer)
        .add_pass(geom_pass_render)
        .set_shader(shader)
        .add_input(RenderPassInput::Buffer(vertex_buffer))
        .add_output(renderer.render_target)
        .add_output(renderer.depth_stencil)
        .build()
        .map_err(|_| Error::CreateRenderPass)?;

    renderer.add_render_pass(render_pass);

    Ok(())
}
