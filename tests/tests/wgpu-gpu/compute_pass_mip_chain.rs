//! A mip chain built by one compute pass after a render pass in the same
//! encoder: each dispatch samples the level above and stores into the next.
//!
//! The render pass leaves the command buffer's tracker knowing only level 0,
//! so it learns the other levels' start states from the compute pass's
//! tracker. Those must be what each level's first dispatch needs (storage),
//! not the state it ends the pass in (sampled): otherwise Vulkan writes the
//! levels in `SHADER_READ_ONLY_OPTIMAL` (VUID-vkCmdDraw-None-09600) and D3D12
//! as a shader resource.

use std::sync::Arc;

use wgpu_test::{gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(COMPUTE_PASS_MIP_CHAIN);
}

const LEVELS: u32 = 3;
const CLEAR: wgpu::Color = wgpu::Color {
    r: 0.0,
    g: 1.0,
    b: 0.0,
    a: 1.0,
};

const SHADER: &str = "
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    textureStore(dst, id.xy, textureLoad(src, id.xy * 2u, 0));
}
";

#[gpu_test]
static COMPUTE_PASS_MIP_CHAIN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(wgpu::DownlevelFlags::COMPUTE_SHADERS)
            .limits(wgpu::Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        let side = 1 << (LEVELS - 1);
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mip chain"),
            size: wgpu::Extent3d {
                width: side,
                height: side,
                depth_or_array_layers: 1,
            },
            mip_level_count: LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let views: Vec<wgpu::TextureView> = (0..LEVELS)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(SHADER.into()),
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
        let groups: Vec<wgpu::BindGroup> = (1..LEVELS as usize)
            .map(|level| {
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&views[level - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&views[level]),
                        },
                    ],
                })
            })
            .collect();

        let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Level 0 drawn first, in the same encoder.
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &views[0],
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(CLEAR),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            for (level, group) in (1..).zip(&groups) {
                pass.set_bind_group(0, group, &[]);
                let n = side >> level;
                pass.dispatch_workgroups(n, n, 1);
            }
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: LEVELS - 1,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: None,
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        ctx.queue.submit([encoder.finish()]);

        let mapped = Arc::new(std::sync::OnceLock::new());
        readback.slice(..).map_async(wgpu::MapMode::Read, {
            let mapped = mapped.clone();
            move |result| mapped.set(result).unwrap()
        });
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        mapped.get().unwrap().as_ref().unwrap();
        let texel = readback.slice(..).get_mapped_range().unwrap()[..4].to_vec();
        assert_eq!(
            texel,
            [0, 255, 0, 255],
            "the last level holds level 0's clear"
        );
    });
