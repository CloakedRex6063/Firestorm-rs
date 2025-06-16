use crate::render_structs::{RenderPass, RenderPassBuilder, RenderPassInput};
use crate::shaders::{geom_ps, geom_vs};
use crate::{BufferHandle, CullMode, Error, FillMode, Format, FrontFace, GraphicsShaderCode, GraphicsShaderCreateInfo, Model, PrimitiveTopology, RenderContext, Renderer, Scissor, TradShaderCode, Viewport};
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use glam::{Mat4, Vec3};

fn geom_pass_render(context: &mut dyn RenderContext, render_pass: &RenderPass, models: &[Model]) {
    let command = context.get_current_frame().command_id;
    let outputs = render_pass.outputs.clone();
    context.bind_shader(command, render_pass.shader);
    for output in &outputs {
        context.clear_color(command, *output, [0f32, 0f32, 0f32, 0f32]);
    }
    context.clear_depth_stencil(command, render_pass.depth.unwrap(), 1.0, 0u8);
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
    struct PerDrawConstants {
        vertex_buffer_index: u32,
    }

    context.bind_render_targets(command, Some(outputs), render_pass.depth);
    
    if let RenderPassInput::Buffer(buffer) = &render_pass.inputs[0] {
        let eye = Vec3::new(0.0, 0.0, -5.0);    
        let target = Vec3::new(0.0, 0.0, 0.0);  
        let up = Vec3::Y;                  

        let view = Mat4::look_at_lh(eye, target, up);
        let proj = Mat4::perspective_lh(60.0f32.to_radians(), 16.0 / 9.0, 0.1, 100.0);
        let vp = proj * view;
        context.write_buffer(*buffer, &vp as *const _ as *const c_void, size_of::<Mat4>());
        context.bind_constant_buffer(command, *buffer, 1);
    }
    
    for model in models {
        context.bind_index_buffer(command, model.index_buffer, 0);
        let push_constant = PerDrawConstants {
            vertex_buffer_index: context.get_gpu_address(model.vertex_buffer),
        };
        
        for mesh in &model.meshes {
            if let RenderPassInput::Buffer(buffer) = &render_pass.inputs[1] {
                let model = Mat4::IDENTITY;
                context.write_buffer(*buffer, &model as *const _ as *const c_void, size_of::<Mat4>());
                context.bind_constant_buffer(command, *buffer, 3);
            }
            context.push_constant(command, &push_constant as *const _ as *const c_void, 1);
            context.draw_indexed(command, mesh.index_count, 1, mesh.index_start, mesh.base_vertex, 0);
        }
    }
    // context.draw(command, 3, 1, 0, 0);
}

pub fn add_geom_pass(renderer: &mut Renderer, global_buffer: BufferHandle, per_draw_buffer: BufferHandle) -> Result<(), Error> {
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
        .add_input(RenderPassInput::Buffer(global_buffer))
        .add_input(RenderPassInput::Buffer(per_draw_buffer))
        .add_output(renderer.render_target)
        .add_output(renderer.depth_stencil)
        .build()
        .map_err(|_| Error::CreateRenderPass)?;

    renderer.add_render_pass(render_pass);

    Ok(())
}
