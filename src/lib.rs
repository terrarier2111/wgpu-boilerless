use anyhow::Error as AnyError;
use bytemuck::Pod;
use parking_lot::lock_api::RwLockReadGuard;
use parking_lot::{RawRwLock, RwLock};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::iter::once;
use std::marker::PhantomData;
use std::num::NonZeroU32;
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
#[cfg(feature = "debug_labels")]
use wgpu::Label;
use wgpu::{
    Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferAddress, BufferUsages, ColorTargetState, CommandEncoder, CommandEncoderDescriptor, CompositeAlphaMode, DepthStencilState, Device, DeviceDescriptor, Dx12Compiler, Extent3d, Features, FragmentState, ImageCopyTexture, ImageDataLayout, Instance, InstanceDescriptor, Limits, MultisampleState, Origin3d, PipelineCompilationOptions, PipelineLayout, PipelineLayoutDescriptor, PowerPreference, PresentMode, PrimitiveState, PushConstantRange, Queue, RenderPass, RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, ShaderModule, ShaderModuleDescriptor, ShaderSource, Surface, SurfaceConfiguration, SurfaceError, Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState
};

pub trait DataSrc {
    fn surface(&self) -> &Surface;

    fn adapter(&self) -> &Adapter;

    fn device(&self) -> &Device;

    fn queue(&self) -> &Queue;

    fn config(&self) -> &RwLock<SurfaceConfiguration>;

    fn set_surface_texture_alive(&self, val: bool);

    fn get_surface_texture_alive(&self) -> bool;
}

pub struct DirectDataSrc {
    surface: Surface<'static>,
    pub adapter: Adapter,
    // can be used by api users to acquire information
    pub device: Device,
    pub queue: Queue,
    config: RwLock<SurfaceConfiguration>,
    // FIXME: should we use a Mutex instead?
    surface_texture_alive: AtomicBool, // FIXME: should we use a Mutex instead cuz it can spin and thus save cycles?
    window_handle: SendSyncPtr<dyn WindowSize>,
}

impl DataSrc for DirectDataSrc {
    #[inline(always)]
    fn surface(&self) -> &Surface {
        &self.surface
    }

    #[inline(always)]
    fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    #[inline(always)]
    fn device(&self) -> &Device {
        &self.device
    }

    #[inline(always)]
    fn queue(&self) -> &Queue {
        &self.queue
    }

    #[inline(always)]
    fn config(&self) -> &RwLock<SurfaceConfiguration> {
        &self.config
    }

    fn set_surface_texture_alive(&self, val: bool) {
        self.surface_texture_alive.store(val, Ordering::Release);
    }

    fn get_surface_texture_alive(&self) -> bool {
        self.surface_texture_alive.load(Ordering::Acquire)
    }
}

impl Drop for DirectDataSrc {
    fn drop(&mut self) {
        unsafe {
            Arc::from_raw(self.window_handle.0);
        } // FIXME: add safety comment
    }
}

pub struct State<D: DataSrc = DirectDataSrc> {
    data_src: D,
}

impl State<DirectDataSrc> {
    /// Tries to create a new `State` from a `StateBuilder`
    ///
    /// returns either the newly created state or an error if
    /// requesting an adapter or device fails.
    #[cfg(not(feature = "custom_data"))]
    pub async fn new(builder: StateBuilder) -> anyhow::Result<Self> {
        use wgpu::{Gles3MinorVersion, InstanceFlags};

        let window = builder
            .window
            .expect("window has to be specified before building the state");
        let size = window.window_size();
        // The instance is a handle to our GPU
        let instance = Instance::new(InstanceDescriptor {
            backends: builder.backends,
            dx12_shader_compiler: Dx12Compiler::Fxc, // TODO: support this!
            flags: InstanceFlags::empty(),           // TODO: support this!
            gles_minor_version: Gles3MinorVersion::Automatic, // TODO: support this!
        }); // used to create adapters and surfaces
        let handle = Arc::into_raw(window);
        let surface = unsafe { instance.create_surface(handle.as_ref().unwrap_unchecked())? }; // FIXME: add safety comment
        let adapter = instance // adapter is a handle to our graphics card
            .request_adapter(&RequestAdapterOptions {
                power_preference: builder.power_pref,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await;
        if let Some(adapter) = adapter {
            let (device, queue) = adapter
                .request_device(
                    &DeviceDescriptor {
                        label: None,
                        required_features: builder.requirements.features,
                        required_limits: builder.requirements.limits,
                        memory_hints: wgpu::MemoryHints::default(), // FIXME: add this as a proper option
                    },
                    None,
                )
                .await?;

            // use the specified format, if none is provided, fallback to the preferred format
            let format = builder
                .format
                .unwrap_or_else(|| surface.get_capabilities(&adapter).formats[0]);
            let config = SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.0,
                height: size.0,
                present_mode: builder.present_mode,
                alpha_mode: builder.alpha_mode,
                view_formats: vec![],             // TODO: support this!
                desired_maximum_frame_latency: 2, // TODO: support this!
            };
            surface.configure(&device, &config);

            return Ok(Self {
                data_src: DirectDataSrc {
                    surface,
                    adapter,
                    device,
                    queue,
                    config: RwLock::new(config),
                    surface_texture_alive: Default::default(),
                    window_handle: SendSyncPtr(handle),
                },
            });
        }
        Err(AnyError::from(NoSuitableAdapterFoundError))
    }
}

impl<D: DataSrc> State<D> {
    /// Creates a new `State` from a `DataSrc`
    #[cfg(feature = "custom_data")]
    pub fn new(data_src: D) -> Self {
        Self { data_src }
    }

    /// Creates a new pipeline layout from its bind group layouts and
    /// its push constant ranges
    pub fn create_pipeline_layout(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        bind_group_layouts: &[&BindGroupLayout],
        push_constant_ranges: &[PushConstantRange],
    ) -> PipelineLayout {
        self.data_src
            .device()
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                #[cfg(feature = "debug_labels")]
                label,
                #[cfg(not(feature = "debug_labels"))]
                label: None,
                bind_group_layouts,
                push_constant_ranges,
            })
    }

    /// Helper method to create a pipeline from its builder
    pub fn create_pipeline(&self, builder: PipelineBuilder) -> RenderPipeline {
        let shaders = builder
            .shader_sources
            .expect("shader sources have to be specified before building the pipeline")
            .into_modules(self);
        let vertex_shader = builder
            .vertex_shader
            .expect("vertex shader has to be specified before building the pipeline");

        self.data_src
            .device()
            .create_render_pipeline(&RenderPipelineDescriptor {
                #[cfg(feature = "debug_labels")]
                label: builder.label,
                #[cfg(not(feature = "debug_labels"))]
                label: None,
                layout: Some(
                    builder
                        .layout
                        .expect("layout has to be specified before building the pipeline"),
                ),
                vertex: VertexState {
                    module: shaders.vertex_module(),
                    entry_point: vertex_shader.entry_point,
                    buffers: vertex_shader.buffers,
                    compilation_options: PipelineCompilationOptions::default() // FIXME: add this as a proper option
                },
                fragment: builder
                    .fragment_shader
                    .map(|fragment_shader| FragmentState {
                        module: shaders.fragment_module(),
                        entry_point: fragment_shader.entry_point,
                        targets: fragment_shader.targets,
                        compilation_options: PipelineCompilationOptions::default() // FIXME: add this as a proper option
                    }),
                primitive: builder.primitive,
                depth_stencil: builder.depth_stencil,
                multisample: builder.multisample,
                multiview: builder.multiview,
                cache: None, // FIXME: add this as a proper option
            })
    }

    /// Creates a shader module from its src
    pub fn create_shader(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        src: ShaderSource,
    ) -> ShaderModule {
        self.data_src
            .device()
            .create_shader_module(ShaderModuleDescriptor {
                #[cfg(feature = "debug_labels")]
                label,
                #[cfg(not(feature = "debug_labels"))]
                label: None,
                source: src,
            })
    }

    /// Returns whether the resizing succeeded or not
    pub fn resize(&self, size: impl Into<(u32, u32)>) -> bool {
        let size = size.into();
        if size.0 > 0 && size.1 > 0 {
            let mut config = self.data_src.config().write();
            config.width = size.0;
            config.height = size.1;
            // FIXME: should we verify that there exist no old textures?
            self.data_src
                .surface()
                .configure(self.data_src.device(), &config);
            true
        } else {
            false
        }
    }

    /// Initiates the rendering process, the passed callback gets
    /// called once all the required state is set up and
    /// once it ran, all the required steps to proceed get executed.
    pub fn render<F: FnOnce(&TextureView, CommandEncoder, &State<D>) -> CommandEncoder>(
        &self,
        callback: F,
        surface_view_desc: &TextureViewDescriptor,
    ) -> Result<(), SurfaceError> {
        self.data_src.set_surface_texture_alive(true);
        let output = self.data_src.surface().get_current_texture()?;
        // get a view of the current texture in order to render on it
        let view = output.texture.create_view(surface_view_desc);

        let encoder = self
            .data_src
            .device()
            .create_command_encoder(&CommandEncoderDescriptor::default());
        // let the user do stuff with the encoder
        let encoder = callback(&view, encoder, self);

        self.data_src.queue().submit(once(encoder.finish()));

        output.present();
        self.data_src.set_surface_texture_alive(false);

        Ok(())
    }

    /// Helper method to create a render pass in the encoder
    pub fn create_render_pass<'a>(
        &self,
        encoder: &'a mut CommandEncoder,
        color_attachments: &'a [Option<RenderPassColorAttachment<'a>>],
        depth_stencil_attachment: Option<RenderPassDepthStencilAttachment<'a>>,
    ) -> RenderPass<'a> {
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None, // TODO: is this needed?
        })
    }

    /// Returns the size defined in the config
    pub fn size(&self) -> (u32, u32) {
        let config = self.data_src.config().read();
        (config.width, config.height)
    }

    /// Returns the surface's current format
    pub fn format(&self) -> TextureFormat {
        let config = self.data_src.config().read();
        config.format
    }

    /// Tries to update the present mode of the surface.
    /// Returns whether update succeeded or not
    pub fn try_update_present_mode(&self, present_mode: PresentMode) -> bool {
        if !self.data_src.get_surface_texture_alive() {
            let mut config = self.data_src.config().write();
            config.present_mode = present_mode;
            self.data_src
                .surface()
                .configure(self.data_src.device(), &config);
            true
        } else {
            false
        }
    }

    /// Updates the present mode of the surface
    /// *Note*: this function will wait until the render call has finished
    pub fn update_present_mode(&self, present_mode: PresentMode) {
        while !self.try_update_present_mode(present_mode) {
            // do nothing, as we just want to update the present mode
        }
    }

    /// Returns a reference to the device
    pub fn device(&self) -> &Device {
        self.data_src.device()
    }

    /// Returns a reference to the queue
    pub fn queue(&self) -> &Queue {
        self.data_src.queue()
    }

    /// Returns a reference to the adapter
    pub fn adapter(&self) -> &Adapter {
        self.data_src.adapter()
    }

    /// SAFETY: The caller has to ensure that the alive state
    /// passed to the function correctly reflects the actual
    /// liveliness of the surface texture
    #[cfg(feature = "custom_data")]
    pub unsafe fn set_surface_texture_alive(&self, alive: bool) {
        self.data_src.set_surface_texture_alive(alive);
    }

    /// Returns a reference to the surface
    #[inline]
    pub fn surface(&self) -> ROSurface {
        ROSurface(self.data_src.surface(), self.data_src.adapter())
    }

    /// Returns a reference to the surface's config.
    /// NOTE: This function is only intended to be used with apis
    pub fn raw_inner_surface_config(&self) -> SurfaceConfigurationRef<'_> {
        SurfaceConfigurationRef(self.data_src.config().read())
    }

    /// Helper method to create a buffer from its content and usage
    pub fn create_buffer<T: Pod>(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        content: &[T],
        usage: BufferUsages,
    ) -> Buffer {
        // FIXME: should we switch from Pod to NoUninit?
        self.data_src
            .device()
            .create_buffer_init(&BufferInitDescriptor {
                #[cfg(feature = "debug_labels")]
                label,
                #[cfg(not(feature = "debug_labels"))]
                label: None,
                contents: bytemuck::cast_slice(content),
                usage,
            })
    }

    /// Helper method to create a texture from its builder
    pub fn create_texture(&self, builder: TextureBuilder) -> Texture {
        // FIXME: reuse create_raw_texture method to reduce code duplication
        let mip_info = builder.mip_info;
        let dimensions = builder
            .inner
            .dimensions
            .expect("dimensions have to be specified before building the texture");
        let texture_size = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: builder.inner.depth_or_array_layers,
        };
        let format = builder
            .inner
            .format
            .expect("format has to be specified before building the texture");
        let diffuse_texture = self.data_src.device().create_texture(&TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: mip_info.mip_level_count,
            sample_count: builder.inner.sample_count,
            dimension: builder
                .inner
                .texture_dimension
                .expect("texture dimension has to be specified before building the texture"),
            // Most images are stored using sRGB so we need to reflect that here.
            format,
            // COPY_DST means that we want to copy data to this texture
            usage: builder.inner.usages.unwrap(),
            #[cfg(feature = "debug_labels")]
            label: builder.inner.label,
            #[cfg(not(feature = "debug_labels"))]
            label: None,
            view_formats: &[], // TODO: support this!
        });
        self.data_src.queue().write_texture(
            // Tells wgpu where to copy the pixel data
            ImageCopyTexture {
                texture: &diffuse_texture,
                mip_level: mip_info.target_mip_level,
                origin: mip_info.origin,
                aspect: builder.aspect,
            },
            // The actual pixel data
            builder
                .data
                .expect("data has to be specified before building the texture"),
            // The layout of the texture
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(
                    format.block_copy_size(Some(builder.aspect)).unwrap() * dimensions.0,
                ),
                rows_per_image: Some(dimensions.1),
            },
            texture_size,
        );

        diffuse_texture
    }

    /// Helper method to create a raw texture from its builder
    pub fn create_raw_texture(&self, builder: RawTextureBuilder) -> Texture {
        let dimensions = builder
            .dimensions
            .expect("dimensions have to be specified before building the texture");
        let texture_size = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            depth_or_array_layers: builder.depth_or_array_layers,
        };
        let format = builder
            .format
            .expect("format has to be specified before building the texture");

        self.data_src.device().create_texture(&TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: builder.mip_level_count,
            sample_count: builder.sample_count,
            dimension: builder
                .texture_dimension
                .expect("texture dimension has to be specified before building the texture"),
            // Most images are stored using sRGB so we need to reflect that here.
            format,
            // COPY_DST means that we want to copy data to this texture
            usage: builder
                .usages
                .expect("texture usages have to be specified before building the texture"),
            #[cfg(feature = "debug_labels")]
            label: builder.label,
            #[cfg(not(feature = "debug_labels"))]
            label: None,
            view_formats: &[], // TODO: support this!
        })
    }

    /// Helper method to create a `BindGroupLayoutEntry` from its entries
    pub fn create_bind_group_layout(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        entries: &[BindGroupLayoutEntry],
    ) -> BindGroupLayout {
        self.data_src
            .device()
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                #[cfg(feature = "debug_labels")]
                label,
                #[cfg(not(feature = "debug_labels"))]
                label: None,
                entries,
            })
    }

    /// Helper method to create a `BindGroup` from its layout and entries
    pub fn create_bind_group(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        layout: &BindGroupLayout,
        entries: &[BindGroupEntry],
    ) -> BindGroup {
        self.data_src
            .device()
            .create_bind_group(&BindGroupDescriptor {
                #[cfg(feature = "debug_labels")]
                label,
                #[cfg(not(feature = "debug_labels"))]
                label: None,
                layout,
                entries,
            })
    }

    /// Helper method to write data to a buffer at a specific offset
    pub fn write_buffer<T: Pod>(&self, buffer: &Buffer, offset: BufferAddress, data: &[T]) {
        self.data_src
            .queue()
            .write_buffer(buffer, offset, bytemuck::cast_slice(data));
    }

    /// Helper method to create a depth texture from its format
    pub fn create_depth_texture(&self, format: TextureFormat) -> Texture {
        let (width, height) = self.size();
        let size = Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture_desc = TextureDescriptor {
            label: None,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        self.data_src.device().create_texture(&texture_desc)
    }
}

#[derive(Default)]
pub struct PipelineBuilder<'a> {
    layout: Option<&'a PipelineLayout>,
    vertex_shader: Option<VertexShaderState<'a>>,
    fragment_shader: Option<FragmentShaderState<'a>>,
    primitive: PrimitiveState,
    depth_stencil: Option<DepthStencilState>,
    multisample: MultisampleState,
    multiview: Option<NonZeroU32>,
    shader_sources: Option<ShaderModuleSources<'a>>,
    #[cfg(feature = "debug_labels")]
    label: Label<'a>,
}

impl<'a> PipelineBuilder<'a> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn layout(mut self, layout: &'a PipelineLayout) -> Self {
        self.layout = Some(layout);
        self
    }

    pub fn vertex(mut self, vertex_shader: VertexShaderState<'a>) -> Self {
        self.vertex_shader = Some(vertex_shader);
        self
    }

    pub fn fragment(mut self, fragment_shader: FragmentShaderState<'a>) -> Self {
        self.fragment_shader = Some(fragment_shader);
        self
    }

    pub fn primitive(mut self, primitive: PrimitiveState) -> Self {
        self.primitive = primitive;
        self
    }

    pub fn depth_stencil(mut self, depth_stencil: DepthStencilState) -> Self {
        self.depth_stencil = Some(depth_stencil);
        self
    }

    /// The default value is `MultisampleState::default()`
    pub fn multisample(mut self, multisample: MultisampleState) -> Self {
        self.multisample = multisample;
        self
    }

    pub fn multiview(mut self, multiview: NonZeroU32) -> Self {
        self.multiview = Some(multiview);
        self
    }

    pub fn shader_src(mut self, shader_sources: ShaderModuleSources<'a>) -> Self {
        // FIXME: do we even need ShaderModuleSources, wouldn't it be cleaner to let the user
        // FIXME: pass in either a Ref to a ShaderModule or a ShaderSource themselves?
        // FIXME: tho this could also make it less nice to use.
        self.shader_sources = Some(shader_sources);
        self
    }

    #[cfg(feature = "debug_labels")]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    #[inline]
    pub fn build<D: DataSrc>(self, state: &State<D>) -> RenderPipeline {
        state.create_pipeline(self)
    }
}

pub struct VertexShaderState<'a> {
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
    /// The format of any vertex buffers used with this pipeline.
    pub buffers: &'a [VertexBufferLayout<'a>],
}

pub struct FragmentShaderState<'a> {
    /// The name of the entry point in the compiled shader. There must be a function that returns
    /// void with this name in the shader.
    pub entry_point: &'a str,
    /// The color state of the render targets.
    pub targets: &'a [Option<ColorTargetState>],
}

pub enum ShaderModuleSources<'a> {
    Single(ModuleSrc<'a>),
    Multi(ModuleSrc<'a>, ModuleSrc<'a>),
}

impl<'a> ShaderModuleSources<'a> {
    fn into_modules<D: DataSrc>(self, state: &'a State<D>) -> ShaderModules<'a> {
        match self {
            ShaderModuleSources::Single(src) => ShaderModules::Single(src.into_module(state)),
            ShaderModuleSources::Multi(vertex_src, fragment_src) => ShaderModules::Multi(
                vertex_src.into_module(state),
                fragment_src.into_module(state),
            ),
        }
    }
}

pub enum ModuleSrc<'a> {
    Source(ShaderSource<'a>, #[cfg(feature = "debug_labels")] Label<'a>),
    Ref(&'a ShaderModule),
}

impl<'a> ModuleSrc<'a> {
    #[cfg(feature = "debug_labels")]
    fn into_module<D: DataSrc>(self, state: &'a State<D>) -> MaybeOwnedModule<'a> {
        match self {
            ModuleSrc::Source(src, label) => {
                MaybeOwnedModule::Owned(state.create_shader(label, src))
            }
            ModuleSrc::Ref(reference) => MaybeOwnedModule::Ref(reference),
        }
    }

    #[cfg(not(feature = "debug_labels"))]
    fn into_module<D: DataSrc>(self, state: &'a State<D>) -> MaybeOwnedModule<'a> {
        match self {
            ModuleSrc::Source(src) => MaybeOwnedModule::Owned(state.create_shader(src)),
            ModuleSrc::Ref(reference) => MaybeOwnedModule::Ref(reference),
        }
    }
}

#[cfg(feature = "debug_labels")]
impl<'a> From<ShaderSource<'a>> for ModuleSrc<'a> {
    #[inline]
    fn from(src: ShaderSource<'a>) -> Self {
        Self::Source(src, None)
    }
}

#[cfg(not(feature = "debug_labels"))]
impl<'a> From<ShaderSource<'a>> for ModuleSrc<'a> {
    #[inline]
    fn from(src: ShaderSource<'a>) -> Self {
        Self::Source(src)
    }
}

impl<'a> From<&'a ShaderModule> for ModuleSrc<'a> {
    #[inline]
    fn from(src: &'a ShaderModule) -> Self {
        Self::Ref(src)
    }
}

enum ShaderModules<'a> {
    Single(MaybeOwnedModule<'a>),
    Multi(MaybeOwnedModule<'a>, MaybeOwnedModule<'a>),
}

impl ShaderModules<'_> {
    fn vertex_module(&self) -> &ShaderModule {
        match self {
            ShaderModules::Single(module) => module.shader_ref(),
            ShaderModules::Multi(vertex_module, _) => vertex_module.shader_ref(),
        }
    }

    fn fragment_module(&self) -> &ShaderModule {
        match self {
            ShaderModules::Single(module) => module.shader_ref(),
            ShaderModules::Multi(_, fragment_module) => fragment_module.shader_ref(),
        }
    }
}

enum MaybeOwnedModule<'a> {
    Owned(ShaderModule),
    Ref(&'a ShaderModule),
}

impl MaybeOwnedModule<'_> {
    fn shader_ref(&self) -> &ShaderModule {
        match self {
            MaybeOwnedModule::Owned(owned) => owned,
            MaybeOwnedModule::Ref(reference) => reference,
        }
    }
}

impl<'a> From<ShaderSource<'a>> for ShaderModuleSources<'a> {
    #[inline]
    fn from(src: ShaderSource<'a>) -> Self {
        Self::Single(ModuleSrc::from(src))
    }
}

impl<'a> From<(ShaderSource<'a>, ShaderSource<'a>)> for ShaderModuleSources<'a> {
    #[inline]
    fn from(src: (ShaderSource<'a>, ShaderSource<'a>)) -> Self {
        Self::Multi(ModuleSrc::from(src.0), ModuleSrc::from(src.1))
    }
}

impl<'a> From<&'a ShaderModule> for ShaderModuleSources<'a> {
    #[inline]
    fn from(src: &'a ShaderModule) -> Self {
        Self::Single(ModuleSrc::from(src))
    }
}

impl<'a> From<(&'a ShaderModule, &'a ShaderModule)> for ShaderModuleSources<'a> {
    #[inline]
    fn from(src: (&'a ShaderModule, &'a ShaderModule)) -> Self {
        Self::Multi(ModuleSrc::from(src.0), ModuleSrc::from(src.1))
    }
}

pub struct RawTextureBuilder<'a> {
    dimensions: Option<(u32, u32)>,
    format: Option<TextureFormat>,
    texture_dimension: Option<TextureDimension>,
    usages: Option<TextureUsages>,
    sample_count: u32,          // we have a default
    mip_level_count: u32,       // we have a default
    depth_or_array_layers: u32, // we have a default
    #[cfg(feature = "debug_labels")]
    label: Label<'a>,
    _phantom_data: PhantomData<&'a ()>, // this is required in case debug_labels are disabled
}

impl Default for RawTextureBuilder<'_> {
    fn default() -> Self {
        Self {
            dimensions: None,
            format: None,
            texture_dimension: None,
            usages: None,
            sample_count: 1,
            mip_level_count: 1,
            depth_or_array_layers: 1,
            #[cfg(feature = "debug_labels")]
            label: None,
            _phantom_data: Default::default(),
        }
    }
}

impl<'a> RawTextureBuilder<'a> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dimensions(mut self, dimensions: (u32, u32)) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn format(mut self, format: TextureFormat) -> Self {
        self.format = Some(format);
        self
    }

    pub fn texture_dimension(mut self, texture_dimension: TextureDimension) -> Self {
        self.texture_dimension = Some(texture_dimension);
        self
    }

    pub fn usages(mut self, usages: TextureUsages) -> Self {
        self.usages = Some(usages);
        self
    }

    /// The default value is 1
    #[inline]
    pub fn sample_count(mut self, sample_count: u32) -> Self {
        self.sample_count = sample_count;
        self
    }

    /// The default value is `1`
    #[inline]
    pub fn mip_level_count(mut self, mip_level_count: u32) -> Self {
        self.mip_level_count = mip_level_count;
        self
    }

    /// The default value is 1
    #[inline]
    pub fn depth_or_array_layers(mut self, depth_or_array_layers: u32) -> Self {
        self.depth_or_array_layers = depth_or_array_layers;
        self
    }

    #[cfg(feature = "debug_labels")]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    #[inline]
    pub fn build<D: DataSrc>(self, state: &State<D>) -> Texture {
        state.create_raw_texture(self)
    }
}

pub struct TextureBuilder<'a> {
    inner: RawTextureBuilder<'a>,
    data: Option<&'a [u8]>,
    aspect: TextureAspect, // we have a default
    mip_info: MipInfo,     // we have a default
}

impl Default for TextureBuilder<'_> {
    fn default() -> Self {
        Self {
            inner: RawTextureBuilder::new()
                .usages(TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST),
            data: None,
            aspect: TextureAspect::All,
            mip_info: MipInfo::default(),
        }
    }
}

impl<'a> TextureBuilder<'a> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn data(mut self, data: &'a [u8]) -> Self {
        self.data = Some(data);
        self
    }

    pub fn dimensions(mut self, dimensions: (u32, u32)) -> Self {
        self.inner = self.inner.dimensions(dimensions);
        self
    }

    pub fn format(mut self, format: TextureFormat) -> Self {
        self.inner = self.inner.format(format);
        self
    }

    pub fn texture_dimension(mut self, texture_dimension: TextureDimension) -> Self {
        self.inner = self.inner.texture_dimension(texture_dimension);
        self
    }

    /// The default value is TextureUsages::TEXTURE_BINDING
    /// *Note*: TextureUsages::COPY_DST gets appended to the usages
    pub fn usages(mut self, usages: TextureUsages) -> Self {
        self.inner = self.inner.usages(usages | TextureUsages::COPY_DST);
        self
    }

    /// The default value is `TextureAspect::All`
    #[inline]
    pub fn aspect(mut self, aspect: TextureAspect) -> Self {
        self.aspect = aspect;
        self
    }

    /// The default value is 1
    #[inline]
    pub fn sample_count(mut self, sample_count: u32) -> Self {
        self.inner = self.inner.sample_count(sample_count);
        self
    }

    /// The default value is `MipInfo::default()`
    #[inline]
    pub fn mip_info(mut self, mip_info: MipInfo) -> Self {
        self.inner.mip_level_count = mip_info.mip_level_count;
        self.mip_info = mip_info;
        self
    }

    /// The default value is 1
    #[inline]
    pub fn depth_or_array_layers(mut self, depth_or_array_layers: u32) -> Self {
        self.inner = self.inner.depth_or_array_layers(depth_or_array_layers);
        self
    }

    #[cfg(feature = "debug_labels")]
    pub fn label(mut self, label: &'a str) -> Self {
        self.label(label);
        self
    }

    #[inline]
    pub fn build<D: DataSrc>(self, state: &State<D>) -> Texture {
        state.create_texture(self)
    }
}

#[cfg(not(feature = "custom_data"))]
pub struct StateBuilder {
    window: Option<Arc<dyn WindowSize>>,
    power_pref: PowerPreference,      // we have a default
    present_mode: PresentMode,        // we have a default
    requirements: DeviceRequirements, // we have a default
    backends: Backends,               // we have a default
    format: Option<TextureFormat>,    // we have a default
    alpha_mode: CompositeAlphaMode,   // we have a default
}

#[cfg(not(feature = "custom_data"))]
impl Default for StateBuilder {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            window: None,
            power_pref: Default::default(),
            present_mode: Default::default(),
            requirements: Default::default(),
            format: None,
            alpha_mode: CompositeAlphaMode::Auto,
        }
    }
}

#[cfg(not(feature = "custom_data"))]
impl StateBuilder {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn window(mut self, window: Arc<dyn WindowSize>) -> Self {
        self.window = Some(window);
        self
    }

    /// The default value is `PowerPreference::LowPower`
    #[inline]
    pub fn power_pref(mut self, power_pref: PowerPreference) -> Self {
        self.power_pref = power_pref;
        self
    }

    /// The default value is `PresentMode::Fifo`
    #[inline]
    pub fn present_mode(mut self, present_mode: PresentMode) -> Self {
        self.present_mode = present_mode;
        self
    }

    /// The default value is `CompositeAlphaMode::Auto`
    #[inline]
    pub fn alpha_mode(mut self, alpha_mode: CompositeAlphaMode) -> Self {
        self.alpha_mode = alpha_mode;
        self
    }

    // FIXME: should we rename this to `requirements`?
    /// The default value is `DeviceRequirements::default()`
    #[inline]
    pub fn device_requirements(mut self, requirements: DeviceRequirements) -> Self {
        self.requirements = requirements;
        self
    }

    /// The default value is `Backends::all()`
    #[inline]
    pub fn backends(mut self, backends: Backends) -> Self {
        self.backends = backends;
        self
    }

    /// The default is to use the preferred format
    /// *Note*: This might be useful when you don't have control over the
    /// pipelines provided to you but you can specify a default.
    /// This method may be deprecated in the future.
    #[inline]
    pub fn format(mut self, format: TextureFormat) -> Self {
        self.format = Some(format);
        self
    }

    #[inline]
    pub async fn build(self) -> anyhow::Result<State> {
        State::new(self).await
    }
}

pub struct MipInfo {
    pub origin: Origin3d,
    pub target_mip_level: u32,
    pub mip_level_count: u32,
}

impl Default for MipInfo {
    fn default() -> Self {
        Self {
            origin: Origin3d::ZERO,
            target_mip_level: 0,
            mip_level_count: 1,
        }
    }
}

#[derive(Default)]
pub struct DeviceRequirements {
    pub features: Features,
    pub limits: Limits,
}

pub struct NoSuitableAdapterFoundError;

impl Debug for NoSuitableAdapterFoundError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str("couldn't create state because no suitable adapter was found")
    }
}

impl Display for NoSuitableAdapterFoundError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str("couldn't create state because no suitable adapter was found")
    }
}

impl Error for NoSuitableAdapterFoundError {}

pub const fn matrix<const COLUMNS: usize>(
    offset: u64,
    location: u32,
    format: VertexFormat,
) -> [VertexAttribute; COLUMNS] {
    let mut ret = [VertexAttribute {
        format,
        offset: 0,
        shader_location: 0,
    }; COLUMNS];

    let mut x = 0;
    while COLUMNS > x {
        ret[x] = VertexAttribute {
            format,
            offset: (offset + format.size() * x as u64) as BufferAddress,
            shader_location: location + x as u32,
        };
        x += 1;
    }

    ret
}

/// This trait is required in order to abstract over the windowing library
///
/// an implementation of this trait could look like this:
/// pub struct WinitWindowWrapper<'a>(&'a Window);
///
/// impl WindowSize for WinitWindowWrapper<'_> {
///
///     fn window_size<T: Into<(u32, u32)>>(&self) -> T {
///         let size = self.0.inner_size();
///         (size.width, size.height)
///     }
///
/// }
///
/// unsafe impl HasRawWindowHandle for WinitWindowWrapper<'_> {
///
///     fn raw_window_handle(&self) -> RawWindowHandle {
///         self.0.raw_window_handle()
///     }
///
/// }
///
pub trait WindowSize: Send + Sync + HasWindowHandle + HasDisplayHandle {
    /// Returns the size of the window in the format (width, height)
    fn window_size(&self) -> (u32, u32);
}

#[cfg(feature = "winit")]
impl WindowSize for winit::window::Window {
    fn window_size(&self) -> (u32, u32) {
        let size = self.inner_size();
        (size.width, size.height)
    }
}

/// this is a read-only version of the Surface struct
pub struct ROSurface<'a>(&'a Surface<'a>, &'a Adapter);

impl ROSurface<'_> {
    /// See [Surface::get_capabilities]
    #[inline]
    pub fn get_supported_formats(&self) -> Vec<TextureFormat> {
        self.0.get_capabilities(self.1).formats
    }

    /// See [Surface::get_capabilities]
    #[inline]
    pub fn get_supported_present_modes(&self) -> Vec<PresentMode> {
        self.0.get_capabilities(self.1).present_modes
    }

    /// See [Surface::get_capabilities]
    #[inline]
    pub fn get_supported_alpha_modes(&self) -> Vec<CompositeAlphaMode> {
        self.0.get_capabilities(self.1).alpha_modes
    }
}

/// this is a wrapper for a RwLockGuard guarding a readable SurfaceConfig
pub struct SurfaceConfigurationRef<'a>(RwLockReadGuard<'a, RawRwLock, SurfaceConfiguration>);

impl AsRef<SurfaceConfiguration> for SurfaceConfigurationRef<'_> {
    fn as_ref(&self) -> &SurfaceConfiguration {
        &self.0
    }
}

impl Deref for SurfaceConfigurationRef<'_> {
    type Target = SurfaceConfiguration;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(transparent)]
struct SendSyncPtr<T: ?Sized>(*const T);

unsafe impl<T: ?Sized> Send for SendSyncPtr<T> {}

unsafe impl<T: ?Sized> Sync for SendSyncPtr<T> {}
