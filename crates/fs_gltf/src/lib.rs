use std::os::raw::c_void;
use fs_renderer::{
    BufferCreateInfo, BufferHandle, Mesh, Model, PrimitiveTopology, Renderer, Vertex,
};
use glam::Mat4;
use gltf::mesh::Mode;
use gltf::{buffer, image};

pub enum Error {
    FileCorrupt,
    NoIndices,
    UploadFailed,
}

pub fn load_gltf(path: &str, renderer: &mut Renderer) -> Result<Model, Error> {
    let import = gltf::import(path).map_err(|_| Error::FileCorrupt)?;
    let document = import.0;
    let buffer_data = import.1;
    let image_data = import.2;

    let mut model = Model {
        vertices: vec![],
        indices: vec![],
        vertex_buffer: BufferHandle::new(0),
        index_buffer: BufferHandle::new(0),
        meshes: vec![],
        materials: vec![],
        textures: vec![],
    };

    for node in document.scenes().flat_map(|s| s.nodes()) {
        traverse_nodes(&mut model, &buffer_data, &image_data, &node, Mat4::IDENTITY);
    }

    let vertex_stride = size_of::<Vertex>() as u32;
    let vertex_length = model.vertices.len() as u32;
    
    let vertex_create_info = BufferCreateInfo {
        stride: vertex_stride,
        elements: vertex_length,
    };
    let vertex_buffer = renderer
        .get_render_context()
        .create_buffer(&vertex_create_info)
        .map_err(|_| Error::FileCorrupt)?;
    
    let index_stride = size_of::<u32>() as u32;
    let index_length = model.indices.len() as u32;
    
    let index_create_info = BufferCreateInfo {
        stride: index_stride,
        elements: index_length,
    };
    let index_buffer = renderer
        .get_render_context()
        .create_buffer(&index_create_info)
        .map_err(|_| Error::FileCorrupt)?;

    model.vertex_buffer = vertex_buffer;
    model.index_buffer = index_buffer;
    
    renderer.get_render_context().write_buffer(
        model.vertex_buffer, 
        model.vertices.as_ptr() as *const c_void,
        (vertex_stride * vertex_length) as usize
    ).map_err(|_| Error::UploadFailed)?;
    
    renderer.get_render_context().write_buffer(
        model.index_buffer, 
        model.indices.as_ptr() as *const c_void,
        (index_stride * index_length) as usize
    ).map_err(|_| Error::UploadFailed)?;

    Ok(model)
}

pub fn traverse_nodes(
    model: &mut Model,
    buffer_data: &Vec<buffer::Data>,
    image_data: &Vec<image::Data>,
    node: &gltf::Node,
    parent_transform: Mat4,
) {
    let world = parent_transform * Mat4::from_cols_array_2d(&node.transform().matrix());
    if let Some(mesh) = node.mesh() {
        for primitive in mesh.primitives() {
            load_primitive(&primitive, model, buffer_data);
        }
    }
    for child in node.children() {
        traverse_nodes(model, buffer_data, image_data, &child, world);
    }
}

pub fn load_primitive(
    primitive: &gltf::Primitive,
    model: &mut Model,
    buffer_data: &Vec<buffer::Data>,
) {
    let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));
    let positions: Vec<[f32; 3]> = reader.read_positions().unwrap().collect();
    let indices: Vec<u32> = reader.read_indices().unwrap().into_u32().collect();

    let normals: Vec<[f32; 3]> = if let Some(temp_normals) = reader.read_normals() {
        temp_normals.collect()
    } else {
        vec![]
    };

    let tex_coords: Vec<[f32; 2]> = if let Some(temp_tex_coords) = reader.read_tex_coords(0) {
        temp_tex_coords.into_f32().collect()
    } else {
        vec![]
    };

    let tangents: Vec<[f32; 4]> = if let Some(temp_tangents) = reader.read_tangents() {
        temp_tangents.collect()
    } else {
        vec![]
    };

    let mut vertices = Vec::with_capacity(positions.len());

    for (i, position) in positions.iter().enumerate() {
        let normal = normals.get(i).cloned().unwrap_or([0f32; 3]);
        let tangent = tangents.get(i).cloned().unwrap_or([0f32; 4]);
        let tex_coord = tex_coords.get(i).cloned().unwrap_or([0f32; 2]);

        let vertex = Vertex {
            position: *position,
            uv_x: tex_coord[0],
            normal,
            uv_y: tex_coord[1],
            tangent,
        };

        vertices.push(vertex);
    }

    let primitive_topology = match primitive.mode() {
        Mode::Points => PrimitiveTopology::PointList,
        Mode::Lines => PrimitiveTopology::LineList,
        Mode::Triangles => PrimitiveTopology::TriangleList,
        _ => todo!(),
    };

    let mesh = Mesh {
        index_start: model.indices.len() as u32,
        index_count: indices.len() as u32,
        base_vertex: vertices.len() as i32,
        material_index: None,
        primitive_topology,
    };

    model.vertices.extend(vertices);
    model.indices.extend(indices);
    model.meshes.push(mesh);
}
