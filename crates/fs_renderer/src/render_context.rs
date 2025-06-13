use std::ffi::c_void;
use crate::*;

pub struct FrameData {
    pub command_id: CommandHandle,
    pub render_target_id: TextureHandle,
    pub fence_value: u64,
}

pub trait RenderContext {
    fn get_current_frame(&mut self) -> &FrameData;
    fn get_mut_current_frame(&mut self) -> &mut FrameData;
    fn get_adapter_info(&self) -> AdapterInfo;
    fn get_texture_format(&self, texture_handle: TextureHandle) -> Format;
    fn get_texture_size(&self, texture_handle: TextureHandle) -> [u32; 2];

    fn create_compute_shader(
        &mut self,
        shader_create_info: &ComputeShaderCreateInfo,
    ) -> Result<ShaderHandle, Error>;
    fn create_graphics_shader(
        &mut self,
        shader_create_info: &GraphicsShaderCreateInfo,
    ) -> Result<ShaderHandle, Error>;
    fn create_texture(
        &mut self,
        texture_create_info: &TextureCreateInfo,
    ) -> Result<TextureHandle, Error>;
    fn create_buffer(
        &mut self,
        buffer_create_info: &BufferCreateInfo,
    ) -> Result<BufferHandle, Error>;
    fn create_command(&mut self, queue_type: QueueType) -> Result<CommandHandle, Error>;
    fn destroy_texture(&mut self, texture_id: TextureHandle);
    fn destroy_buffer(&mut self, buffer_id: BufferHandle);

    fn write_buffer(&mut self, buffer_id: BufferHandle, data: *const c_void, len: usize) -> Result<(), Error>;
    fn get_gpu_address(&mut self, buffer_id: BufferHandle) -> u32;

    fn begin_command(&mut self, command_id: CommandHandle);
    fn end_command(&mut self, command_id: CommandHandle);
    fn submit_command(&mut self, command_id: CommandHandle, queue_type: QueueType);
    fn present(&mut self, vsync: bool);
    fn clear_color(
        &mut self,
        command_handle: CommandHandle,
        texture_handle: TextureHandle,
        color: [f32; 4],
    );
    fn transition_resource(
        &mut self,
        command_handle: CommandHandle,
        resource_handle: ResourceHandle,
        new_state: ResourceState,
    );
    fn bind_render_targets(
        &mut self,
        command_handle: CommandHandle,
        render_target: Option<Vec<TextureHandle>>,
        depth_stencil_handle: Option<TextureHandle>,
    );
    fn bind_shader(&mut self, command_handle: CommandHandle, shader_handle: ShaderHandle);
    fn set_viewport(&mut self, command_handle: CommandHandle, viewport: Viewport);
    fn set_scissor(&mut self, command_handle: CommandHandle, scissor: Scissor);
    fn push_constant(&mut self, command_id: CommandHandle, data: *const c_void, count: u32);
    fn set_primitive_topology(
        &mut self,
        command_handle: CommandHandle,
        topology: PrimitiveTopology,
    );
    fn draw(
        &mut self,
        command_handle: CommandHandle,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    );
    fn copy_to_swapchain(
        &mut self,
        command_handle: CommandHandle,
        render_target_handle: TextureHandle,
    );
}
