/*!
  Use the gpu to run computations
*/
use std::convert::TryInto;
use wgpu::util::DeviceExt;

///***execute_gpu_sync***
///
///Sync version of `execute_gpu`
pub fn execute_gpu_sync(items: Vec<Vec<u32>>, spirv: &[u32]) -> Vec<u32> {
    futures::executor::block_on(execute_gpu(items, spirv))
}
///***execute_gpu***
///
///Takes a vector of buffers, the first buffer is used as the output buffer.
///
///Takes the compiled spirv as well and run the compiled operations.
///
///The user must ensure that the spirv and the vector have the same number of buffers.
pub async fn execute_gpu(items: Vec<Vec<u32>>, spirv: &[u32]) -> Vec<u32> {
    //the first item is the output
    let len = items[0].len();

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    let cs_module = device.create_shader_module(wgpu::ShaderModuleSource::SpirV(spirv.into()));

    let slice_size = len * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    let mut bind_groups = vec![];
    let mut bind_group_layouts = vec![];
    let mut storage_buffers = vec![];

    for b in items {
        let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Storage Buffer"),
            contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsage::STORAGE
                | wgpu::BufferUsage::COPY_DST
                | wgpu::BufferUsage::COPY_SRC,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    readonly: false,
                    dynamic: false,
                    min_binding_size: wgpu::BufferSize::new(4),
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(storage_buffer.slice(..)),
            }],
        });
        bind_groups.push(bind_group);
        bind_group_layouts.push(bind_group_layout);
        storage_buffers.push(storage_buffer);
    }
    let bind_group_layouts: Vec<&wgpu::BindGroupLayout> = bind_group_layouts.iter().collect();

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &bind_group_layouts,
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        for (i, bind_group) in bind_groups.iter().enumerate() {
            cpass.set_bind_group(i as u32, &bind_group, &[]);
        }
        cpass.dispatch(len as u32, 1, 1);
    }

    //output
    encoder.copy_buffer_to_buffer(&storage_buffers[0], 0, &staging_buffer, 0, size);

    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);

    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();

        let result = data
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        drop(data);
        staging_buffer.unmap();

        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}

//Rexport shaderc for the calc macro
#[doc(hidden)]
pub use regex;
#[doc(hidden)]
pub use shaderc;
///***calc***
///
///Macro to compile spirv from an equation.
///
///All buffers need to have the name b with a digit like b1.
///
///b0 is the output buffer.
///
///Examples:
///```
///  calc!(b0 = 4)
///  calc!(b0 += 4)
///  calc!(b0 += b1 * b2 + 5)
///```
#[macro_export]
macro_rules! calc {
    // b0 += b1
    // b0 += 4
    // b0 *= b1 + b2
    ($e:expr) => {{
        let string = stringify!($e);
        let n = string.match_indices("b").count();
        let string = gpu::regex::Regex::new(r#"(b\d+*)"#)
            .unwrap()
            .replace_all(string, "$1.data[idx]");

        let mut source = String::new();
        source.push_str("#version 450\n");
        for i in 0..n {
            source.push_str(&format!(
                "layout(set = {}, binding = 0) buffer Data{} {{uint data[];}} b{};\n",
                i, i, i
            ));
        }
        source.push_str(&format!(
            "void main() {{
            uint idx = gl_GlobalInvocationID.x;
            {};
            }}",
            string
        ));

        let mut compiler = gpu::shaderc::Compiler::new().unwrap();
        let artifact = compiler
            .compile_into_spirv(
                &source,
                gpu::shaderc::ShaderKind::Compute,
                "shader.glsl",
                "main",
                None,
            )
            .unwrap();

        artifact
    }};
}
