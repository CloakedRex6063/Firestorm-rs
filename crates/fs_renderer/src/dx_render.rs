use crate::render_context::*;
use crate::*;
use fs_window::Window;
use log::*;
use std::cmp::PartialEq;
use std::ffi::{CStr, c_void};
use std::mem::{ManuallyDrop, MaybeUninit};
use std::ptr::null_mut;
use windows::Win32::Foundation::{HANDLE, RECT};
use windows::Win32::Graphics::Direct3D::{
    D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_LINELIST, D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
    D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob,
};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_D32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC, DXGI_FORMAT_R32_UINT};
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventA, INFINITE, WaitForSingleObject};
use windows::core::{Interface, PCSTR};

impl From<Format> for DXGI_FORMAT {
    fn from(value: Format) -> Self {
        match value {
            Format::RGBA8UNORM => DXGI_FORMAT_R8G8B8A8_UNORM,
            Format::RGBA16 => DXGI_FORMAT_R16G16B16A16_UNORM,
            Format::D32 => DXGI_FORMAT_D32_FLOAT,
            _ => DXGI_FORMAT_UNKNOWN,
        }
    }
}

impl From<CullMode> for D3D12_CULL_MODE {
    fn from(value: CullMode) -> Self {
        match value {
            CullMode::None => D3D12_CULL_MODE_NONE,
            CullMode::Front => D3D12_CULL_MODE_FRONT,
            CullMode::Back => D3D12_CULL_MODE_BACK,
        }
    }
}

impl From<FrontFace> for bool {
    fn from(value: FrontFace) -> Self {
        match value {
            FrontFace::Clockwise => false,
            FrontFace::CounterClockwise => true,
        }
    }
}

impl From<FillMode> for D3D12_FILL_MODE {
    fn from(value: FillMode) -> Self {
        match value {
            FillMode::Solid => D3D12_FILL_MODE_SOLID,
            FillMode::Wireframe => D3D12_FILL_MODE_WIREFRAME,
        }
    }
}

impl From<PrimitiveTopology> for D3D12_PRIMITIVE_TOPOLOGY_TYPE {
    fn from(value: PrimitiveTopology) -> Self {
        match value {
            PrimitiveTopology::PointList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
            PrimitiveTopology::LineList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
            PrimitiveTopology::TriangleList => D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        }
    }
}

impl From<ResourceState> for D3D12_RESOURCE_STATES {
    fn from(value: ResourceState) -> Self {
        match value {
            ResourceState::Common => D3D12_RESOURCE_STATE_COMMON,
            ResourceState::RenderTarget => D3D12_RESOURCE_STATE_RENDER_TARGET,
            ResourceState::DepthStencil => D3D12_RESOURCE_STATE_DEPTH_WRITE,
            ResourceState::ShaderResource => D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE,
            ResourceState::TransferSrc => D3D12_RESOURCE_STATE_COPY_SOURCE,
            ResourceState::TransferDst => D3D12_RESOURCE_STATE_COPY_DEST,
        }
    }
}

struct Command {
    command_list: ID3D12GraphicsCommandList4,
    command_allocator: ID3D12CommandAllocator,
}

struct Shader {
    pipeline: ID3D12PipelineState,
}

struct DescriptorHeap {
    heap: ID3D12DescriptorHeap,
    #[cfg(debug_assertions)]
    capacity: u32,
    at: u32,
    stride: u32,
    cpu_start: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu_start: D3D12_GPU_DESCRIPTOR_HANDLE,
    free_list: Vec<u32>,
}

impl DescriptorHeap {
    fn allocate(&mut self) -> Descriptor {
        let mut index = self.at;

        #[cfg(debug_assertions)]
        if index >= self.capacity {
            panic!("Out of descriptors");
        }

        if !self.free_list.is_empty() {
            index = self.free_list.pop().unwrap();
        } else {
            self.at += 1;
        }
        Descriptor {
            cpu: D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.cpu_start.ptr + self.stride as usize * index as usize,
            },
            gpu: D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: self.gpu_start.ptr + self.stride as u64 * index as u64,
            },
            index,
        }
    }
}

struct Resource {
    resource: ID3D12Resource,
    state: ResourceState,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Descriptor {
    cpu: D3D12_CPU_DESCRIPTOR_HANDLE,
    gpu: D3D12_GPU_DESCRIPTOR_HANDLE,
    index: u32,
}

struct Texture {
    resource_handle: ResourceHandle,
    rtv_descriptor: Option<Descriptor>,
    dsv_descriptor: Option<Descriptor>,
    cbv_srv_uav_descriptor: Option<Descriptor>,
    format: Format,
    size: [u32; 2],
}

struct Buffer {
    resource_handle: ResourceHandle,
    cbv_srv_uav_descriptor: Descriptor,
    size: u32,
}

pub struct RenderContextDX {
    device: ID3D12Device14,
    adapter: IDXGIAdapter4,
    swapchain: IDXGISwapChain4,
    graphics_queue: ID3D12CommandQueue,
    compute_queue: ID3D12CommandQueue,
    transfer_queue: ID3D12CommandQueue,
    transfer_fence: ID3D12Fence,
    graphics_fence: ID3D12Fence,
    compute_fence: ID3D12Fence,
    fence_event: HANDLE,

    root_signature: ID3D12RootSignature,

    rtv_heap: DescriptorHeap,
    dsv_heap: DescriptorHeap,
    cbv_srv_uav_heap: DescriptorHeap,

    frame_datas: [MaybeUninit<FrameData>; 3],
    current_frame: usize,
    commands: Vec<Command>,
    free_commands: Vec<u32>,
    shaders: Vec<Shader>,
    free_shaders: Vec<u32>,
    textures: Vec<Texture>,
    free_textures: Vec<u32>,
    buffers: Vec<Buffer>,
    free_buffers: Vec<u32>,

    resources: Vec<Resource>,
    free_resources: Vec<u32>,
}

#[derive(PartialEq, Debug)]
pub enum DescriptorHeapType {
    CbvSrvUav,
    Rtv,
    Dsv,
}

impl RenderContextDX {
    pub fn new(window: &Window) -> Result<Box<dyn RenderContext>, Error> {
        let factory = create_factory()?;
        let adapter = choose_adapter(&factory)?;
        #[cfg(debug_assertions)]
        let debug_context = create_debug_context()?;
        #[cfg(debug_assertions)]
        enable_debug_layer(&debug_context);
        let device = create_device(&adapter)?;
        #[cfg(debug_assertions)]
        setup_debug_callback(&device)?;
        let graphics_queue = create_command_queue(&device, QueueType::Graphics)?;
        let compute_queue = create_command_queue(&device, QueueType::Compute)?;
        let transfer_queue = create_command_queue(&device, QueueType::Transfer)?;
        let swapchain = create_swapchain(window, &factory, &graphics_queue)?;
        let rtv_heap = create_descriptor_heap(&device, DescriptorHeapType::Rtv, 64)?;
        let dsv_heap = create_descriptor_heap(&device, DescriptorHeapType::Dsv, 64)?;
        let cbv_srv_uav_heap =
            create_descriptor_heap(&device, DescriptorHeapType::CbvSrvUav, 4096)?;
        let transfer_fence = create_fence(&device)?;
        let graphics_fence = create_fence(&device)?;
        let compute_fence = create_fence(&device)?;
        let fence_event = create_fence_event()?;

        let root_signature = create_root_signature(&device)?;

        let textures = vec![];
        let free_textures = vec![];
        let commands = vec![];
        let free_commands = vec![];
        let buffers = vec![];
        let free_buffers = vec![];
        let shaders = vec![];
        let free_shaders = vec![];
        let resources = vec![];
        let free_resources = vec![];

        let mut render_context = Self {
            device,
            adapter,
            swapchain,
            graphics_queue,
            compute_queue,
            transfer_queue,
            transfer_fence,
            graphics_fence,
            compute_fence,
            fence_event,
            root_signature,
            rtv_heap,
            dsv_heap,
            cbv_srv_uav_heap,
            frame_datas: unsafe { MaybeUninit::uninit().assume_init() },
            current_frame: 0,
            commands,
            free_commands,
            shaders,
            free_shaders,
            textures,
            free_textures,
            buffers,
            free_buffers,
            resources,
            free_resources,
        };

        let frame_datas = create_frame_data(&mut render_context, &window)?;

        render_context.frame_datas = frame_datas.map(|frame_data| MaybeUninit::new(frame_data));

        Ok(Box::new(render_context))
    }
}

impl RenderContext for RenderContextDX {
    fn get_current_frame(&mut self) -> &FrameData {
        unsafe { self.frame_datas[self.current_frame].assume_init_ref() }
    }

    fn get_mut_current_frame(&mut self) -> &mut FrameData {
        unsafe { self.frame_datas[self.current_frame].assume_init_mut() }
    }

    fn get_adapter_info(&self) -> AdapterInfo {
        let desc = unsafe { self.adapter.GetDesc3() }.unwrap();

        let name = String::from_utf16_lossy(&desc.Description);
        AdapterInfo {
            name,
            vram: desc.DedicatedVideoMemory as u64,
            sys_mem: desc.SharedSystemMemory as u64,
        }
    }

    fn get_texture_format(&self, texture_handle: TextureHandle) -> Format {
        let texture = &self.textures[texture_handle.0 as usize];
        texture.format
    }

    fn get_texture_size(&self, texture_handle: TextureHandle) -> [u32; 2] {
        let texture = &self.textures[texture_handle.0 as usize];
        texture.size
    }

    fn create_compute_shader(
        &mut self,
        shader_create_info: &ComputeShaderCreateInfo,
    ) -> Result<ShaderHandle, Error> {
        let pipeline =
            create_compute_pipeline(&self.device, &self.root_signature, &shader_create_info)?;
        let shader = Shader { pipeline };
        Ok(add_shader(
            shader,
            &mut self.shaders,
            &mut self.free_shaders,
        ))
    }

    fn create_graphics_shader(
        &mut self,
        shader_create_info: &GraphicsShaderCreateInfo,
    ) -> Result<ShaderHandle, Error> {
        let pipeline =
            create_graphics_pipeline(&self.device, &self.root_signature, &shader_create_info)?;
        let shader = Shader { pipeline };
        Ok(add_shader(
            shader,
            &mut self.shaders,
            &mut self.free_shaders,
        ))
    }

    fn create_texture(
        &mut self,
        texture_create_info: &TextureCreateInfo,
    ) -> Result<TextureHandle, Error> {
        let flags = texture_create_info.texture_flags;
        let dx_resource = create_resource(&self.device, &texture_create_info)
            .map_err(|_| Error::CreateTexture)?;

        let rtv_descriptor = if flags.contains(TextureFlags::RenderTarget) {
            Some(create_render_target_view(
                &self.device,
                &mut self.rtv_heap,
                &texture_create_info,
                &dx_resource,
            ))
        } else {
            None
        };
        let dsv_descriptor = if flags.contains(TextureFlags::DepthStencil) {
            Some(create_depth_stencil_view(
                &self.device,
                &mut self.dsv_heap,
                &texture_create_info,
                &dx_resource,
            ))
        } else {
            None
        };
        let cbv_srv_uav_descriptor = if flags.contains(TextureFlags::ShaderResource) {
            Some(create_shader_resource_view(
                &self.device,
                &mut self.cbv_srv_uav_heap,
                &texture_create_info,
                &dx_resource,
            ))
        } else {
            None
        };
        let format = texture_create_info.format;

        let resource = Resource {
            resource: dx_resource,
            state: ResourceState::Common,
        };
        let resource_handle = add_resource(resource, &mut self.resources, &mut self.free_resources);
        let size = [texture_create_info.width, texture_create_info.height];

        let texture = Texture {
            resource_handle,
            rtv_descriptor,
            dsv_descriptor,
            cbv_srv_uav_descriptor,
            format,
            size,
        };
        Ok(add_texture(
            texture,
            &mut self.textures,
            &mut self.free_textures,
        ))
    }

    fn create_buffer(
        &mut self,
        buffer_create_info: &BufferCreateInfo,
    ) -> Result<BufferHandle, Error> {
        let dx_resource = create_buffer_resource(&self.device, &buffer_create_info)
            .map_err(|_| Error::CreateBuffer)?;

        let cbv_srv_uav_descriptor = create_buffer_srv(
            &self.device,
            &mut self.cbv_srv_uav_heap,
            &buffer_create_info,
            &dx_resource,
        );
        let resource = Resource {
            resource: dx_resource,
            state: ResourceState::Common,
        };
        let resource_handle = add_resource(resource, &mut self.resources, &mut self.free_resources);

        debug!(
            "Created buffer with {} element(s) and stride {}",
            buffer_create_info.elements, buffer_create_info.stride
        );
        
        let size = buffer_create_info.stride * buffer_create_info.elements;
        
        let buffer = Buffer {
            resource_handle,
            cbv_srv_uav_descriptor,
            size,
        };
        Ok(add_buffer(
            buffer,
            &mut self.buffers,
            &mut self.free_buffers,
        ))
    }
    fn create_command(&mut self, _: QueueType) -> Result<CommandHandle, Error> {
        let command = create_command(&self.device).map_err(|_| Error::CreateCommand)?;
        let command_handle = if self.free_commands.is_empty() {
            self.commands.push(command);
            CommandHandle::new((self.commands.len() - 1) as u32)
        } else {
            CommandHandle::new(self.free_commands.pop().unwrap())
        };
        Ok(command_handle)
    }

    fn destroy_texture(&mut self, texture_id: TextureHandle) {
        todo!()
    }

    fn destroy_buffer(&mut self, buffer_id: BufferHandle) {
        todo!()
    }

    fn write_buffer(
        &mut self,
        buffer_id: BufferHandle,
        data: *const c_void,
        len: usize,
    ) -> Result<(), Error> {
        let buffer = &self.buffers[buffer_id.0 as usize];
        let resource = &self.resources[buffer.resource_handle.0 as usize];
        unsafe {
            let mut data_ptr = null_mut();
            resource
                .resource
                .Map(0, None, Some(&mut data_ptr))
                .map_err(|_| Error::MapBuffer)?;

            if !data_ptr.is_null() {
                std::ptr::copy_nonoverlapping(data, data_ptr, len);
            }

            resource.resource.Unmap(0, None);
        }
        Ok(())
    }

    fn get_gpu_address(&mut self, buffer_id: BufferHandle) -> u32 {
        let buffer = &self.buffers[buffer_id.0 as usize];
        buffer.cbv_srv_uav_descriptor.index
    }

    fn begin_command(&mut self, command_id: CommandHandle) {
        unsafe {
            let command = &self.commands[command_id.0 as usize];
            let command_list = &command.command_list;
            command.command_allocator.Reset().unwrap();
            command_list
                .Reset(&command.command_allocator, None)
                .unwrap();
            command_list.SetGraphicsRootSignature(&self.root_signature);
            command_list.SetComputeRootSignature(&self.root_signature);
            command_list.SetDescriptorHeaps(&[Some(self.cbv_srv_uav_heap.heap.clone())])
        }
    }

    fn end_command(&mut self, command_id: CommandHandle) {
        unsafe {
            let texture = self.get_current_frame().render_target_id;
            let render_target = &self.textures[texture.0 as usize];
            self.transition_resource(
                command_id,
                render_target.resource_handle,
                ResourceState::Common,
            );
            let command = &self.commands[command_id.0 as usize].command_list;
            command.Close().unwrap();
        }
    }

    fn submit_command(&mut self, command_id: CommandHandle, queue_type: QueueType) {
        let command = &self.commands[command_id.0 as usize].command_list;
        let lists = [Some(command.cast::<ID3D12CommandList>().unwrap())];
        unsafe {
            match queue_type {
                QueueType::Graphics => {
                    self.graphics_queue.ExecuteCommandLists(&lists);
                }
                QueueType::Compute => {
                    self.compute_queue.ExecuteCommandLists(&lists);
                }
                QueueType::Transfer => {
                    self.transfer_queue.ExecuteCommandLists(&lists);
                }
            }
        }
    }

    fn present(&mut self, vsync: bool) {
        unsafe {
            let _ = self.swapchain.Present(vsync as u32, DXGI_PRESENT(0)); // TODO return result
            let fence_value = self.get_current_frame().fence_value;

            let _ = self
                .graphics_queue
                .Signal(&self.graphics_fence, fence_value);

            self.current_frame = self.swapchain.GetCurrentBackBufferIndex() as usize;

            let next_fence_value = self.get_current_frame().fence_value;

            if self.graphics_fence.GetCompletedValue() < next_fence_value {
                self.graphics_fence
                    .SetEventOnCompletion(next_fence_value, self.fence_event)
                    .unwrap();
                WaitForSingleObject(self.fence_event, INFINITE);
            }
            self.get_mut_current_frame().fence_value = fence_value + 1;
        }
    }

    fn clear_color(
        &mut self,
        command_handle: CommandHandle,
        texture_handle: TextureHandle,
        color: [f32; 4],
    ) {
        let command = self.commands[command_handle.0 as usize]
            .command_list
            .clone();
        let texture = &self.textures[texture_handle.0 as usize];
        let descriptor = texture.rtv_descriptor.unwrap().cpu.clone();
        self.transition_resource(
            command_handle,
            texture.resource_handle,
            ResourceState::RenderTarget,
        );
        unsafe {
            command.ClearRenderTargetView(descriptor, &color, None);
        }
    }

    fn clear_depth_stencil(
        &mut self,
        command_handle: CommandHandle,
        depth_stencil_handle: TextureHandle,
        depth: f32,
        stencil: u8,
    ) {
        let command = self.commands[command_handle.0 as usize]
            .command_list
            .clone();
        let depth_stencil = &self.textures[depth_stencil_handle.0 as usize];
        let descriptor = depth_stencil.dsv_descriptor.unwrap().cpu.clone();
        self.transition_resource(
            command_handle,
            depth_stencil.resource_handle,
            ResourceState::DepthStencil,
        );
        unsafe {
            command.ClearDepthStencilView(descriptor, D3D12_CLEAR_FLAG_DEPTH, depth, stencil, None);
        }
    }

    fn transition_resource(
        &mut self,
        command_handle: CommandHandle,
        resource_handle: ResourceHandle,
        new_state: ResourceState,
    ) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        let resource = &mut self.resources[resource_handle.0 as usize];
        if resource.state != new_state {
            let barrier = D3D12_RESOURCE_BARRIER {
                Type: Default::default(),
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: ManuallyDrop::new(Some(resource.resource.clone())),
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: resource.state.into(),
                        StateAfter: new_state.into(),
                    }),
                },
            };
            unsafe {
                command.ResourceBarrier(&[barrier]);
            }
            resource.state = new_state;
        }
    }

    fn bind_render_targets(
        &mut self,
        command_handle: CommandHandle,
        render_target_handles: Option<Vec<TextureHandle>>,
        depth_stencil_handle: Option<TextureHandle>,
    ) {
        let command = self.commands[command_handle.0 as usize]
            .command_list
            .clone();

        let mut rtvs: [MaybeUninit<D3D12_CPU_DESCRIPTOR_HANDLE>; 8] =
            unsafe { MaybeUninit::uninit().assume_init() };
        let mut length = 0;

        let render_targets = if let Some(handles) = &render_target_handles {
            for (i, handle) in handles.iter().enumerate() {
                debug_assert!(i < rtvs.len());

                let resource = self.textures[handle.0 as usize].resource_handle;
                self.transition_resource(command_handle, resource, ResourceState::RenderTarget);

                rtvs[i].write(
                    self.textures[handle.0 as usize]
                        .rtv_descriptor
                        .unwrap()
                        .cpu
                        .clone(),
                );
                length += 1;
            }
            Some(rtvs.as_ptr() as *const D3D12_CPU_DESCRIPTOR_HANDLE)
        } else {
            None
        };

        let descriptor: D3D12_CPU_DESCRIPTOR_HANDLE;
        let depth_stencil = if depth_stencil_handle.is_some() {
            let depth_stencil = &self.textures[depth_stencil_handle.unwrap().0 as usize];
            descriptor = depth_stencil.dsv_descriptor.unwrap().cpu.clone();
            self.transition_resource(
                command_handle,
                depth_stencil.resource_handle,
                ResourceState::DepthStencil,
            );
            Some(&descriptor as *const D3D12_CPU_DESCRIPTOR_HANDLE)
        } else {
            None
        };

        unsafe {
            command.OMSetRenderTargets(length as u32, render_targets, false, depth_stencil);
        }
    }

    fn bind_shader(&mut self, command_handle: CommandHandle, shader_handle: ShaderHandle) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        let shader = &self.shaders[shader_handle.0 as usize];
        unsafe {
            command.SetPipelineState(&shader.pipeline);
        }
    }

    fn bind_index_buffer(&mut self, command_handle: CommandHandle, buffer_handle: BufferHandle, offset: u64) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        let buffer = &self.buffers[buffer_handle.0 as usize];
        let resource = &self.resources[buffer.resource_handle.0 as usize];
        unsafe {
            let view = D3D12_INDEX_BUFFER_VIEW{
                BufferLocation: resource.resource.GetGPUVirtualAddress() + offset,
                SizeInBytes: buffer.size,
                Format: DXGI_FORMAT_R32_UINT,
            };
            let ptr = Some(&view as *const _);
            command.IASetIndexBuffer(ptr);
        }
    }

    fn bind_constant_buffer(&mut self, command_handle: CommandHandle, buffer_handle: BufferHandle, index: u32) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        let buffer = &self.buffers[buffer_handle.0 as usize];
        let resource = &self.resources[buffer.resource_handle.0 as usize];
        unsafe {
            command.SetGraphicsRootConstantBufferView(index, resource.resource.GetGPUVirtualAddress());
        }
    }

    fn set_viewport(&mut self, command_handle: CommandHandle, viewport: Viewport) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        let dx_viewport = D3D12_VIEWPORT {
            TopLeftX: viewport.offset[0],
            TopLeftY: viewport.offset[1],
            Width: viewport.dimensions[0],
            Height: viewport.dimensions[1],
            MinDepth: viewport.depth_range[0],
            MaxDepth: viewport.depth_range[1],
        };
        unsafe {
            command.RSSetViewports(&[dx_viewport]);
        }
    }

    fn set_scissor(&mut self, command_handle: CommandHandle, scissor: Scissor) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        let dx_rect = RECT {
            left: scissor.min[0],
            top: scissor.min[1],
            right: scissor.max[0],
            bottom: scissor.max[1],
        };
        unsafe {
            command.RSSetScissorRects(&[dx_rect]);
        }
    }

    fn push_constant(&mut self, command_id: CommandHandle, data: *const c_void, count: u32) {
        let command = &self.commands[command_id.0 as usize].command_list;
        unsafe {
            command.SetGraphicsRoot32BitConstants(0, count, data, 0);
        }
    }

    fn set_primitive_topology(
        &mut self,
        command_handle: CommandHandle,
        topology: PrimitiveTopology,
    ) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        unsafe {
            let dx_topology = match topology {
                PrimitiveTopology::PointList => D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
                PrimitiveTopology::LineList => D3D_PRIMITIVE_TOPOLOGY_LINELIST,
                PrimitiveTopology::TriangleList => D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            };
            command.IASetPrimitiveTopology(dx_topology);
        }
    }

    fn draw(
        &mut self,
        command_handle: CommandHandle,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        unsafe {
            command.DrawInstanced(vertex_count, instance_count, first_vertex, first_instance);
        }
    }

    fn draw_indexed(&mut self, command_handle: CommandHandle, index_count: u32, instance_count: u32, first_index: u32, vertex_offset: i32, first_instance: u32) {
        let command = &self.commands[command_handle.0 as usize].command_list;
        unsafe {
            command.DrawIndexedInstanced(index_count, instance_count, first_index, vertex_offset, first_instance);
        }
    }

    fn copy_to_swapchain(
        &mut self,
        command_handle: CommandHandle,
        render_target_handle: TextureHandle,
    ) {
        let command = self.commands[command_handle.0 as usize]
            .command_list
            .clone();
        let swapchain_handle = self.get_current_frame().render_target_id;
        let swapchain = self.textures[swapchain_handle.0 as usize].resource_handle;
        let render_target = self.textures[render_target_handle.0 as usize].resource_handle;
        let swap_size = self.textures[swapchain_handle.0 as usize].size;
        let render_size = self.textures[render_target.0 as usize].size;
        self.transition_resource(command_handle, render_target, ResourceState::TransferSrc);
        self.transition_resource(command_handle, swapchain, ResourceState::TransferDst);

        let swapchain_resource = &self.resources[swapchain.0 as usize];
        let render_target_resource = &self.resources[render_target.0 as usize];

        if render_size == swap_size {
            unsafe {
                command.CopyResource(
                    &swapchain_resource.resource,
                    &render_target_resource.resource,
                );
            }
        } else {
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(swapchain_resource.resource.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: ManuallyDrop::new(Some(render_target_resource.resource.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            unsafe {
                command.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            }
        }
    }
}

fn create_factory() -> Result<IDXGIFactory7, Error> {
    unsafe {
        #[cfg(debug_assertions)]
        let flags = DXGI_CREATE_FACTORY_DEBUG;
        #[cfg(not(debug_assertions))]
        let flags = DXGI_CREATE_FACTORY_FLAGS::default();
        let factory: IDXGIFactory7 = CreateDXGIFactory2(flags).map_err(|_| Error::CreateFactory)?;
        debug!("Created DXGI Factory");
        Ok(factory)
    }
}

fn choose_adapter(factory: &IDXGIFactory7) -> Result<IDXGIAdapter4, Error> {
    unsafe {
        let adapter: IDXGIAdapter4 = factory
            .EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            .map_err(|_| Error::GetAdapter)?;
        adapter
            .GetDesc3()
            .map(|desc| {
                let description = String::from_utf16_lossy(
                    &desc
                        .Description
                        .iter()
                        .take_while(|&&c| c != 0)
                        .cloned()
                        .collect::<Vec<u16>>(),
                );
                debug!("Adapter: {}", description);
                let dedicated_video_memory =
                    desc.DedicatedVideoMemory as f32 / 1024.0 / 1024.0 / 1024.0;
                let shared_system_memory =
                    desc.SharedSystemMemory as f32 / 1024.0 / 1024.0 / 1024.0;
                debug!("Dedicated VRAM: {} GB", dedicated_video_memory);
                debug!("Shared System Memory: {} GB", shared_system_memory);
                debug!(
                    "Total VRAM: {} GB",
                    dedicated_video_memory + shared_system_memory
                );
            })
            .map_err(|_| Error::GetAdapter)?;
        Ok(adapter)
    }
}

fn create_device(adapter: &IDXGIAdapter4) -> Result<ID3D12Device14, Error> {
    unsafe {
        let mut device: Option<ID3D12Device14> = None;
        D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device)
            .map_err(|_| Error::CreateFactory)?;
        debug!("Created D3D12 Device");
        Ok(device.unwrap())
    }
}

unsafe extern "system" fn debug_callback(
    _: D3D12_MESSAGE_CATEGORY,
    severity: D3D12_MESSAGE_SEVERITY,
    _: D3D12_MESSAGE_ID,
    pdescription: PCSTR,
    _: *mut core::ffi::c_void,
) {
    let message = unsafe { CStr::from_ptr(pdescription.0 as *const i8).to_string_lossy() };
    match severity {
        D3D12_MESSAGE_SEVERITY_ERROR => error!("{}", message),
        D3D12_MESSAGE_SEVERITY_WARNING => warn!("{}", message),
        D3D12_MESSAGE_SEVERITY_INFO => info!("{}", message),
        D3D12_MESSAGE_SEVERITY_MESSAGE => debug!("Message: {}", message),
        _ => {}
    }
}

#[cfg(debug_assertions)]
fn create_debug_context() -> Result<ID3D12Debug4, Error> {
    unsafe {
        let mut debug_context: Option<ID3D12Debug4> = None;
        D3D12GetDebugInterface(&mut debug_context).map_err(|_| Error::CreateDebugContext)?;
        debug!("Created D3D12 Debug Context");
        Ok(debug_context.unwrap())
    }
}

#[cfg(debug_assertions)]
fn enable_debug_layer(debug_context: &ID3D12Debug4) {
    unsafe {
        debug_context.EnableDebugLayer();
        debug_context.SetEnableGPUBasedValidation(true);
        debug_context.SetEnableSynchronizedCommandQueueValidation(true);
        debug!("Enabled Debug Layer");
    }
}

#[cfg(debug_assertions)]
fn setup_debug_callback(device: &ID3D12Device14) -> Result<(), Error> {
    unsafe {
        let info_queue: ID3D12InfoQueue = device.cast().map_err(|_| Error::EnableDebugLayer)?;
        let info_queue1: ID3D12InfoQueue1 =
            info_queue.cast().map_err(|_| Error::EnableDebugLayer)?;
        let mut cookie: u32 = 0;
        info_queue1
            .RegisterMessageCallback(
                Some(debug_callback),
                D3D12_MESSAGE_CALLBACK_FLAG_NONE,
                null_mut(),
                &mut cookie,
            )
            .map_err(|_| Error::EnableDebugLayer)?;
        debug!("Registered Debug Callback");
        Ok(())
    }
}

fn create_swapchain(
    window: &Window,
    factory: &IDXGIFactory7,
    command_queue: &ID3D12CommandQueue,
) -> Result<IDXGISwapChain4, Error> {
    unsafe {
        let size = window.get_size();
        let desc = DXGI_SWAP_CHAIN_DESC1 {
            Width: size.x as u32,
            Height: size.y as u32,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            Stereo: Default::default(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_BACK_BUFFER,
            BufferCount: 3,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
            AlphaMode: Default::default(),
            Flags: DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING.0 as u32,
        };
        let hwnd = window.get_handle().clone();
        let swapchain: IDXGISwapChain1 = factory
            .CreateSwapChainForHwnd(command_queue, hwnd, &desc, None, None::<&IDXGIOutput>)
            .map_err(|_| Error::CreateSwapchain)?;
        let swapchain4: IDXGISwapChain4 = swapchain.cast().map_err(|_| Error::CreateSwapchain)?;
        debug!("Created Swapchain");
        Ok(swapchain4)
    }
}

fn create_command_queue(
    device: &ID3D12Device14,
    queue_type: QueueType,
) -> Result<ID3D12CommandQueue, Error> {
    unsafe {
        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: Default::default(),
            Priority: 0,
            Flags: Default::default(),
            NodeMask: 0,
        };
        let command_queue = device
            .CreateCommandQueue(&queue_desc)
            .map_err(|_| Error::CreateCommandQueue)?;
        debug!("Create {:?} command queue", queue_type);
        Ok(command_queue)
    }
}

fn create_descriptor_heap(
    device: &ID3D12Device14,
    heap_type: DescriptorHeapType,
    num_descriptors: u32,
) -> Result<DescriptorHeap, Error> {
    unsafe {
        let mut flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        if heap_type == DescriptorHeapType::CbvSrvUav {
            flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        }

        let dx_heap_type: D3D12_DESCRIPTOR_HEAP_TYPE = match heap_type {
            DescriptorHeapType::CbvSrvUav => D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            DescriptorHeapType::Rtv => D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
            DescriptorHeapType::Dsv => D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
        };
        let stride = device.GetDescriptorHandleIncrementSize(dx_heap_type);

        let heap_desc = D3D12_DESCRIPTOR_HEAP_DESC {
            Type: dx_heap_type,
            NumDescriptors: num_descriptors,
            Flags: flags,
            NodeMask: 0,
        };
        let heap: ID3D12DescriptorHeap = device
            .CreateDescriptorHeap(&heap_desc)
            .map_err(|_| Error::CreateDescriptorHeap)?;

        let cpu_start = heap.GetCPUDescriptorHandleForHeapStart();

        let gpu_start = if flags.contains(D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE) {
            heap.GetGPUDescriptorHandleForHeapStart()
        } else {
            D3D12_GPU_DESCRIPTOR_HANDLE { ptr: 0 }
        };

        debug!("Created {:?} Descriptor Heap", heap_type);
        let descriptor_heap = DescriptorHeap {
            heap,
            #[cfg(debug_assertions)]
            capacity: num_descriptors,
            at: 0,
            stride,
            cpu_start,
            gpu_start,
            free_list: vec![],
        };
        Ok(descriptor_heap)
    }
}

fn create_fence(device: &ID3D12Device14) -> Result<ID3D12Fence, Error> {
    unsafe {
        let fence = device
            .CreateFence(0, D3D12_FENCE_FLAG_NONE)
            .map_err(|_| Error::CreateFence)?;
        Ok(fence)
    }
}

fn create_fence_event() -> Result<HANDLE, Error> {
    unsafe {
        let fence_event = CreateEventA(None, false, false, None).map_err(|_| Error::CreateFence);
        fence_event
    }
}

fn create_command(device: &ID3D12Device14) -> Result<Command, Error> {
    unsafe {
        let command_type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        let command_allocator = device
            .CreateCommandAllocator(command_type)
            .map_err(|_| Error::CreateCommand)?;
        let command_list: ID3D12GraphicsCommandList4 = device
            .CreateCommandList(
                0,
                command_type,
                &command_allocator,
                None::<&ID3D12PipelineState>,
            )
            .map_err(|_| Error::CreateCommand)?;
        command_list.Close().map_err(|_| Error::CreateCommand)?;

        debug!("Created Command");

        Ok(Command {
            command_list,
            command_allocator,
        })
    }
}

fn create_frame_data(
    render_context_dx: &mut RenderContextDX,
    window: &Window,
) -> Result<[FrameData; 3], Error> {
    let size = window.get_size();
    let create_info = TextureCreateInfo {
        width: size.x as u32,
        height: size.y as u32,
        array_layers: 1,
        mip_levels: 1,
        format: Format::RGBA8UNORM,
        texture_flags: TextureFlags::RenderTarget.into(),
        data: vec![],
    };

    let mut frame_datas = [(); 3].map(|_| None);

    for (i, frame) in frame_datas.iter_mut().enumerate() {
        unsafe {
            let back_buffer: ID3D12Resource = render_context_dx
                .swapchain
                .GetBuffer(i as u32)
                .map_err(|_| Error::GetBackBuffer)?;
            let device = &render_context_dx.device;
            let rtv_heap = &mut render_context_dx.rtv_heap;
            let descriptor =
                create_render_target_view(device, rtv_heap, &create_info, &back_buffer);
            let resources = &mut render_context_dx.resources;
            let free_resources = &mut render_context_dx.free_resources;
            let textures = &mut render_context_dx.textures;
            let free_textures = &mut render_context_dx.free_textures;
            let commands = &mut render_context_dx.commands;
            let free_commands = &mut render_context_dx.free_commands;

            let size = [window.get_size().x as u32, window.get_size().y as u32];

            let texture_handle: u32 = if free_textures.is_empty() {
                let texture = Texture {
                    resource_handle: add_resource(
                        Resource {
                            resource: back_buffer,
                            state: ResourceState::Common,
                        },
                        resources,
                        free_resources,
                    ),
                    rtv_descriptor: Some(descriptor),
                    dsv_descriptor: None,
                    cbv_srv_uav_descriptor: None,
                    format: Format::RGBA8UNORM,
                    size,
                };
                textures.push(texture);
                (textures.len() - 1) as u32
            } else {
                let texture = Texture {
                    resource_handle: add_resource(
                        Resource {
                            resource: back_buffer,
                            state: ResourceState::Common,
                        },
                        resources,
                        free_resources,
                    ),
                    rtv_descriptor: Some(descriptor),
                    dsv_descriptor: None,
                    cbv_srv_uav_descriptor: None,
                    format: Format::RGBA8UNORM,
                    size,
                };

                let index = free_textures.pop().unwrap();
                textures[index as usize] = texture;
                index
            };

            let command = create_command(device).map_err(|_| Error::CreateCommand)?;
            let command_handle = if free_commands.is_empty() {
                commands.push(command);
                CommandHandle::new((commands.len() - 1) as u32)
            } else {
                CommandHandle::new(free_commands.pop().unwrap())
            };

            *frame = Some(FrameData {
                command_id: command_handle,
                render_target_id: TextureHandle::new(texture_handle),
                fence_value: 0,
            });
        }
    }

    debug!("Created Frame Data");

    Ok(frame_datas.map(|opt| opt.unwrap()))
}

fn add_resource(
    resource: Resource,
    resources: &mut Vec<Resource>,
    free_resources: &mut Vec<u32>,
) -> ResourceHandle {
    if free_resources.is_empty() {
        resources.push(resource);
        ResourceHandle::new((resources.len() - 1) as u32)
    } else {
        resources[free_resources.pop().unwrap() as usize] = resource;
        ResourceHandle::new(free_resources.pop().unwrap())
    }
}

fn add_texture(
    texture: Texture,
    textures: &mut Vec<Texture>,
    free_textures: &mut Vec<u32>,
) -> TextureHandle {
    if free_textures.is_empty() {
        textures.push(texture);
        TextureHandle::new((textures.len() - 1) as u32)
    } else {
        textures[free_textures.pop().unwrap() as usize] = texture;
        TextureHandle::new(free_textures.pop().unwrap())
    }
}

fn add_buffer(
    buffer: Buffer,
    buffers: &mut Vec<Buffer>,
    free_buffers: &mut Vec<u32>,
) -> BufferHandle {
    if free_buffers.is_empty() {
        buffers.push(buffer);
        BufferHandle::new((buffers.len() - 1) as u32)
    } else {
        buffers[free_buffers.pop().unwrap() as usize] = buffer;
        BufferHandle::new(free_buffers.pop().unwrap())
    }
}

fn add_shader(
    shader: Shader,
    shaders: &mut Vec<Shader>,
    free_shaders: &mut Vec<u32>,
) -> ShaderHandle {
    if free_shaders.is_empty() {
        shaders.push(shader);
        ShaderHandle::new((shaders.len() - 1) as u32)
    } else {
        shaders[free_shaders.pop().unwrap() as usize] = shader;
        ShaderHandle::new(free_shaders.pop().unwrap())
    }
}

fn create_resource(
    device: &ID3D12Device14,
    texture_create_info: &TextureCreateInfo,
) -> Result<ID3D12Resource, Error> {
    unsafe {
        let heap_flags = D3D12_HEAP_FLAG_NONE;
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };

        let format: DXGI_FORMAT;
        let clear_value = match texture_create_info.format {
            Format::RGBA8UNORM => {
                format = DXGI_FORMAT_R8G8B8A8_UNORM;
                D3D12_CLEAR_VALUE {
                    Format: format,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        Color: [0.0, 0.0, 0.0, 0.0],
                    },
                }
            }
            Format::RGBA16 => {
                format = DXGI_FORMAT_R16G16B16A16_UNORM;
                D3D12_CLEAR_VALUE {
                    Format: format,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        Color: [0.0, 0.0, 0.0, 0.0],
                    },
                }
            }
            Format::D32 => {
                format = DXGI_FORMAT_D32_FLOAT;
                D3D12_CLEAR_VALUE {
                    Format: format,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                            Depth: 1.0,
                            Stencil: 0,
                        },
                    },
                }
            }
            _ => {
                format = DXGI_FORMAT_UNKNOWN;
                D3D12_CLEAR_VALUE {
                    Format: DXGI_FORMAT_UNKNOWN,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        Color: [0.0, 0.0, 0.0, 0.0],
                    },
                }
            }
        };

        let mut flags = D3D12_RESOURCE_FLAG_NONE;

        let texture_flags = texture_create_info.texture_flags;

        if texture_flags.contains(TextureFlags::RenderTarget) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        }

        if texture_flags.contains(TextureFlags::DepthStencil) {
            flags |= D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
        }

        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: texture_create_info.width as u64,
            Height: texture_create_info.height,
            DepthOrArraySize: texture_create_info.array_layers,
            MipLevels: texture_create_info.mip_levels,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: flags,
        };

        let mut resource: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &heap_properties,
                heap_flags,
                &resource_desc,
                D3D12_RESOURCE_STATE_COMMON,
                Some(&clear_value),
                &mut resource,
            )
            .map_err(|_| Error::CreateTexture)?;
        Ok(resource.unwrap())
    }
}

fn create_buffer_resource(
    device: &ID3D12Device14,
    buffer_create_info: &BufferCreateInfo,
) -> Result<ID3D12Resource, Error> {
    unsafe {
        let heap_flags = D3D12_HEAP_FLAG_NONE;
        let heap_properties = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 0,
            VisibleNodeMask: 0,
        };
        let resource_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Alignment: 0,
            Width: (buffer_create_info.elements * buffer_create_info.stride) as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: Default::default(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: Default::default(),
        };
        let mut resource: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &heap_properties,
                heap_flags,
                &resource_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
            .map_err(|_| Error::CreateTexture)?;

        Ok(resource.unwrap())
    }
}

fn create_render_target_view(
    device: &ID3D12Device14,
    heap: &mut DescriptorHeap,
    texture_create_info: &TextureCreateInfo,
    resource: &ID3D12Resource,
) -> Descriptor {
    let descriptor = heap.allocate();
    let format = texture_create_info.format;
    let rtv_desc = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: format.into(),
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE2D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_RTV {
                MipSlice: 0,
                PlaneSlice: 0,
            },
        },
    };
    unsafe {
        device.CreateRenderTargetView(resource, Some(&rtv_desc), descriptor.cpu);
    }
    descriptor
}

fn create_depth_stencil_view(
    device: &ID3D12Device14,
    heap: &mut DescriptorHeap,
    texture_create_info: &TextureCreateInfo,
    resource: &ID3D12Resource,
) -> Descriptor {
    let descriptor = heap.allocate();
    let format = texture_create_info.format;
    let dsv_desc = D3D12_DEPTH_STENCIL_VIEW_DESC {
        Format: format.into(),
        ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
        Flags: Default::default(),
        Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_DSV { MipSlice: 0 },
        },
    };
    unsafe {
        device.CreateDepthStencilView(resource, Some(&dsv_desc), descriptor.cpu);
    }
    descriptor
}

fn create_shader_resource_view(
    device: &ID3D12Device14,
    heap: &mut DescriptorHeap,
    texture_create_info: &TextureCreateInfo,
    resource: &ID3D12Resource,
) -> Descriptor {
    let descriptor = heap.allocate();
    let format = texture_create_info.format;
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format.into(),
        ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Texture2D: D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            },
        },
    };
    unsafe {
        device.CreateShaderResourceView(resource, Some(&srv_desc), descriptor.cpu);
    }
    descriptor
}

fn create_buffer_srv(
    device: &ID3D12Device14,
    heap: &mut DescriptorHeap,
    buffer_create_info: &BufferCreateInfo,
    resource: &ID3D12Resource,
) -> Descriptor {
    let descriptor = heap.allocate();
    let format = DXGI_FORMAT_UNKNOWN;
    let srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
        Format: format.into(),
        ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
        Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
        Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
            Buffer: D3D12_BUFFER_SRV {
                FirstElement: 0,
                NumElements: buffer_create_info.elements,
                StructureByteStride: buffer_create_info.stride,
                Flags: Default::default(),
            },
        },
    };
    unsafe {
        device.CreateShaderResourceView(resource, Some(&srv_desc), descriptor.cpu);
    }
    descriptor
}

fn create_root_signature(device: &ID3D12Device14) -> Result<ID3D12RootSignature, Error> {
    let root_params: [D3D12_ROOT_PARAMETER1; 4] = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            ShaderVisibility: Default::default(),
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 58, // 64 dwords, 6 are used for the constant buffers
                },
            },
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            ShaderVisibility: Default::default(),
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            ShaderVisibility: Default::default(),
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 2,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
        },
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_CBV,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: 3,
                    RegisterSpace: 0,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE,
                },
            },
            ShaderVisibility: Default::default(),
        },
    ];

    let sampler = D3D12_STATIC_SAMPLER_DESC1 {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressV: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        AddressW: D3D12_TEXTURE_ADDRESS_MODE_WRAP,
        MipLODBias: 0.0,
        MaxAnisotropy: 0,
        ComparisonFunc: Default::default(),
        BorderColor: Default::default(),
        MinLOD: 0.0,
        MaxLOD: 13.0,
        ShaderRegister: 0,
        RegisterSpace: 0,
        ShaderVisibility: Default::default(),
        Flags: Default::default(),
    };

    let sig_desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_2,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_2: D3D12_ROOT_SIGNATURE_DESC2 {
                NumParameters: 4,
                pParameters: root_params.as_ptr(),
                NumStaticSamplers: 1,
                pStaticSamplers: &sampler as *const D3D12_STATIC_SAMPLER_DESC1,
                Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
            },
        },
    };
    let mut signature_blob: Option<ID3DBlob> = None;
    let mut error_blob: Option<ID3DBlob> = None;

    unsafe {
        let version_result = D3D12SerializeVersionedRootSignature(
            &sig_desc,
            &mut signature_blob,
            Some(&mut error_blob),
        )
        .map_err(|_| Error::CreateRootSignature);

        if let Some(error) = error_blob {
            let ptr = error.GetBufferPointer() as *const u8;
            let size = error.GetBufferSize();
            let slice = std::slice::from_raw_parts(ptr, size);
            let error_str = std::str::from_utf8(slice).unwrap();
            error!("Error: {}", error_str);
        }

        version_result?;

        let blob_ptr = signature_blob.as_ref().unwrap().GetBufferPointer();
        let blob_len = signature_blob.as_ref().unwrap().GetBufferSize();
        let blob = std::slice::from_raw_parts(blob_ptr as *const u8, blob_len);

        let signature: ID3D12RootSignature = device
            .CreateRootSignature(0, blob)
            .map_err(|_| Error::CreateRootSignature)?;

        debug!("Created Root Signature");

        Ok(signature)
    }
}

fn create_compute_pipeline(
    device: &ID3D12Device14,
    root_sig: &ID3D12RootSignature,
    shader_create_info: &ComputeShaderCreateInfo,
) -> Result<ID3D12PipelineState, Error> {
    unsafe {
        let bytecode = D3D12_SHADER_BYTECODE {
            pShaderBytecode: shader_create_info.shader_code.as_ptr() as _,
            BytecodeLength: shader_create_info.shader_code.len(),
        };
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            CS: bytecode,
            NodeMask: 0,
            CachedPSO: Default::default(),
            Flags: Default::default(),
        };
        Ok(device
            .CreateComputePipelineState(&desc)
            .map_err(|_| Error::CreatePipeline)?)
    }
}

fn get_bytecode(code: &Vec<u8>) -> D3D12_SHADER_BYTECODE {
    D3D12_SHADER_BYTECODE {
        pShaderBytecode: code.as_ptr() as _,
        BytecodeLength: code.len(),
    }
}

fn create_graphics_pipeline(
    device: &ID3D12Device14,
    root_sig: &ID3D12RootSignature,
    shader_create_info: &GraphicsShaderCreateInfo,
) -> Result<ID3D12PipelineState, Error> {
    unsafe {
        let mut vs: D3D12_SHADER_BYTECODE = Default::default();
        let mut vertex_code: Vec<u8> = Vec::new();
        let mut ps: D3D12_SHADER_BYTECODE = Default::default();
        let mut pixel_code: Vec<u8> = Vec::new();
        let mut ds: D3D12_SHADER_BYTECODE = Default::default();
        let mut domain_code: Vec<u8> = Vec::new();
        let mut hs: D3D12_SHADER_BYTECODE = Default::default();
        let mut hull_code: Vec<u8> = Vec::new();
        let mut gs: D3D12_SHADER_BYTECODE = Default::default();
        let mut geom_code: Vec<u8> = Vec::new();
        let mut ms: D3D12_SHADER_BYTECODE = Default::default();
        let mut mesh_code: Vec<u8> = Vec::new();
        let mut ts: D3D12_SHADER_BYTECODE = Default::default();
        let mut task_code: Vec<u8> = Vec::new();
        match &shader_create_info.graphics_shader_code {
            GraphicsShaderCode { trad_shader_code } => {
                if trad_shader_code.vertex_code.is_some() {
                    vertex_code = trad_shader_code.vertex_code.clone().unwrap();
                    vs = get_bytecode(&vertex_code);
                } else {
                    Err(Error::CreatePipeline)?
                }
                if trad_shader_code.hull_code.is_some() {
                    hull_code = trad_shader_code.hull_code.clone().unwrap();
                    hs = get_bytecode(&hull_code);
                }
                if trad_shader_code.geom_code.is_some() {
                    geom_code = trad_shader_code.geom_code.clone().unwrap();
                    gs = get_bytecode(&geom_code);
                }
                if trad_shader_code.domain_code.is_some() {
                    domain_code = trad_shader_code.domain_code.clone().unwrap();
                    ds = get_bytecode(&domain_code);
                }
            }
            GraphicsShaderCode { mesh_shader_code } => {
                if mesh_shader_code.mesh_code.is_some() {
                    mesh_code = mesh_shader_code.mesh_code.clone().unwrap();
                    ms = get_bytecode(&mesh_code);
                }
                if mesh_shader_code.task_code.is_some() {
                    task_code = mesh_shader_code.task_code.clone().unwrap();
                    ts = get_bytecode(&task_code);
                }
            }
        }
        pixel_code = shader_create_info.pixel_code.clone().unwrap();
        ps = get_bytecode(&pixel_code);

        let rtv_formats = shader_create_info.rtv_formats.clone();

        let mut rtv_format_array: [DXGI_FORMAT; 8] = [DXGI_FORMAT_UNKNOWN; 8];
        for (i, val) in rtv_formats.into_iter().take(8).enumerate() {
            rtv_format_array[i] = val.into();
        }

        let blend_descs = rtv_format_array.map(|format| D3D12_RENDER_TARGET_BLEND_DESC {
            BlendEnable: Default::default(),
            LogicOpEnable: Default::default(),
            SrcBlend: Default::default(),
            DestBlend: Default::default(),
            BlendOp: Default::default(),
            SrcBlendAlpha: Default::default(),
            DestBlendAlpha: Default::default(),
            BlendOpAlpha: Default::default(),
            LogicOp: Default::default(),
            RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
        });

        let blend_state = D3D12_BLEND_DESC {
            AlphaToCoverageEnable: Default::default(),
            IndependentBlendEnable: Default::default(),
            RenderTarget: blend_descs,
        };

        let ccw: bool = shader_create_info.front_face.into();

        let rasterizer_state = D3D12_RASTERIZER_DESC {
            FillMode: shader_create_info.fill_mode.into(),
            CullMode: shader_create_info.cull_mode.into(),
            FrontCounterClockwise: ccw.into(),
            DepthBias: 0,
            DepthBiasClamp: 0.0,
            SlopeScaledDepthBias: 0.0,
            DepthClipEnable: Default::default(),
            MultisampleEnable: false.into(),
            AntialiasedLineEnable: false.into(),
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        };

        let depth_stencil_state = D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: true.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
            DepthFunc: D3D12_COMPARISON_FUNC_LESS,
            StencilEnable: false.into(),
            StencilReadMask: 0,
            StencilWriteMask: 0,
            FrontFace: Default::default(),
            BackFace: Default::default(),
        };

        let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: ManuallyDrop::new(Some(root_sig.clone())),
            VS: vs,
            PS: ps,
            DS: ds,
            HS: hs,
            GS: gs,
            StreamOutput: Default::default(),
            BlendState: blend_state,
            SampleMask: D3D12_DEFAULT_SAMPLE_MASK,
            RasterizerState: rasterizer_state,
            DepthStencilState: depth_stencil_state,
            InputLayout: Default::default(),
            IBStripCutValue: Default::default(),
            PrimitiveTopologyType: shader_create_info.primitive_topology.clone().into(),
            NumRenderTargets: shader_create_info.rtv_formats.len() as u32,
            RTVFormats: rtv_format_array,
            DSVFormat: shader_create_info.dsv_format.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            NodeMask: 0,
            CachedPSO: Default::default(),
            Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
        };
        Ok(device
            .CreateGraphicsPipelineState(&desc)
            .map_err(|_| Error::CreatePipeline)?)
    }
}
