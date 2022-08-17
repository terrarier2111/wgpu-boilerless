use anyhow::Error as AnyError;
use bytemuck::Pod;
use parking_lot::RwLock;
use raw_window_handle::HasRawWindowHandle;
use std::error::Error;
use std::fmt::{Debug, Display, Formatter};
use std::iter::once;
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, Ordering};
use wgpu::util::{BufferInitDescriptor, DeviceExt};
#[cfg(feature = "debug_labels")]
use wgpu::Label;
use wgpu::{
    Adapter, Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferAddress, BufferUsages,
    ColorTargetState, CommandEncoder, CommandEncoderDescriptor, DepthStencilState, Device,
    DeviceDescriptor, Extent3d, Features, FragmentState, ImageCopyTexture, ImageDataLayout,
    Instance, Limits, MultisampleState, Origin3d, PipelineLayout, PipelineLayoutDescriptor,
    PowerPreference, PresentMode, PrimitiveState, PushConstantRange, Queue, RenderPass,
    RenderPassColorAttachment, RenderPassDepthStencilAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, ShaderModule,
    ShaderModuleDescriptor, ShaderSource, Surface, SurfaceConfiguration, SurfaceError, Texture,
    TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexAttribute, VertexBufferLayout, VertexFormat, VertexState,
};

pub struct State {
    surface: Surface,
    adapter: Adapter, // can be used by api users to acquire information
    device: Device,
    queue: Queue,
    config: RwLock<SurfaceConfiguration>, // FIXME: should we use a Mutex instead?
    surface_texture_alive: AtomicBool,
}

impl State {
    /// Tries to create a new `State` from a `StateBuilder`
    ///
    /// returns either the newly created state or an error if
    /// requesting an adapter or device fails.
    pub async fn new<T: WindowSize>(builder: StateBuilder<T>) -> anyhow::Result<Self> {
        let window = builder
            .window
            .expect("window has to be specified before building the state");
        let size = window.window_size();
        // The instance is a handle to our GPU
        let instance = Instance::new(builder.backends); // used to create adapters and surfaces
        let surface = unsafe { instance.create_surface(&window) };
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
                        features: builder.requirements.features,
                        limits: builder.requirements.limits,
                    },
                    None,
                )
                .await?;

            // use the specified format, if none is provided, fallback to the preferred format
            let format = builder
                .format
                .unwrap_or_else(|| surface.get_supported_formats(&adapter)[0]);
            let config = SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format,
                width: size.0,
                height: size.0,
                present_mode: builder.present_mode,
            };
            surface.configure(&device, &config);

            return Ok(Self {
                surface,
                adapter,
                device,
                queue,
                config: RwLock::new(config),
                surface_texture_alive: Default::default(),
            });
        }
        Err(AnyError::from(NoSuitableAdapterFoundError))
    }

    /// Creates a new pipeline layout from its bind group layouts and
    /// its push constant ranges
    pub fn create_pipeline_layout(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        bind_group_layouts: &[&BindGroupLayout],
        push_constant_ranges: &[PushConstantRange],
    ) -> PipelineLayout {
        self.device
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
            .to_modules(self);
        let vertex_shader = builder
            .vertex_shader
            .expect("vertex shader has to be specified before building the pipeline");

        self.device
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
                },
                fragment: builder
                    .fragment_shader
                    .map(|fragment_shader| FragmentState {
                        module: shaders.fragment_module(),
                        entry_point: fragment_shader.entry_point,
                        targets: fragment_shader.targets,
                    }),
                primitive: builder
                    .primitive
                    .expect("primitive has to be specified before building the pipeline"),
                depth_stencil: builder.depth_stencil,
                multisample: builder.multisample,
                multiview: builder.multiview,
            })
    }

    /// Creates a shader module from its src
    pub fn create_shader(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        src: ShaderSource,
    ) -> ShaderModule {
        self.device.create_shader_module(ShaderModuleDescriptor {
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
            let mut config = self.config.write();
            config.width = size.0;
            config.height = size.1;
            // FIXME: should we verify that there exist no old textures?
            self.surface.configure(&self.device, &*config);
            true
        } else {
            false
        }
    }

    /// Initiates the rendering process, the passed callback gets
    /// called once all the required state is set up and
    /// once it ran, all the required steps to proceed get executed.
    pub fn render<F: FnOnce(&TextureView, CommandEncoder, &State) -> CommandEncoder>(
        &self,
        callback: F,
        surface_view_desc: &TextureViewDescriptor,
    ) -> Result<(), SurfaceError> {
        self.surface_texture_alive.store(true, Ordering::Release);
        let output = self.surface.get_current_texture()?;
        // get a view of the current texture in order to render on it
        let view = output.texture.create_view(surface_view_desc);

        let encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor::default());
        // let the user do stuff with the encoder
        let encoder = callback(&view, encoder, self);

        self.queue.submit(once(encoder.finish()));

        output.present();
        self.surface_texture_alive.store(false, Ordering::Release);

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
        })
    }

    /// Returns the size defined in the config
    pub fn size(&self) -> (u32, u32) {
        let config = self.config.read();
        (config.width, config.height)
    }

    /// Returns the format defined in the config
    pub fn format(&self) -> TextureFormat {
        let config = self.config.read();
        config.format.clone()
    }

    /// Tries to update the present mode of the surface.
    /// Returns whether update succeeded or not
    pub fn try_update_present_mode(&self, present_mode: PresentMode) -> bool {
        if !self.surface_texture_alive.load(Ordering::Acquire) {
            let mut config = self.config.write();
            config.present_mode = present_mode;
            self.surface.configure(&self.device, &*config);
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
    #[inline(always)]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the queue
    #[inline(always)]
    pub const fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns a reference to the surface
    #[inline(always)]
    pub const fn surface(&self) -> &Surface {
        &self.surface
    }

    /// Returns a reference to the adapter which can be used to
    /// acquire information
    #[inline(always)]
    pub const fn adapter(&self) -> &Adapter {
        &self.adapter
    }

    /// Helper method to create a buffer from its content and usage
    pub fn create_buffer<T: Pod>(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        content: &[T],
        usage: BufferUsages,
    ) -> Buffer {
        // FIXME: should we switch from Pod to NoUninit?
        self.device.create_buffer_init(&BufferInitDescriptor {
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
        let mip_info = builder.mip_info;
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
        let diffuse_texture = self.device.create_texture(&TextureDescriptor {
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            size: texture_size,
            mip_level_count: mip_info.mip_level_count, // We'll talk about this a little later
            sample_count: builder.sample_count,
            dimension: builder
                .texture_dimension
                .expect("texture dimension has to be specified before building the texture"),
            // Most images are stored using sRGB so we need to reflect that here.
            format,
            // COPY_DST means that we want to copy data to this texture
            usage: builder.usages,
            #[cfg(feature = "debug_labels")]
            label: builder.label,
            #[cfg(not(feature = "debug_labels"))]
            label: None,
        });
        self.queue.write_texture(
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
                bytes_per_row: NonZeroU32::new(format.describe().block_size as u32 * dimensions.0),
                rows_per_image: NonZeroU32::new(dimensions.1),
            },
            texture_size,
        );

        diffuse_texture
    }

    /// Helper method to create a `BindGroupLayoutEntry` from its entries
    pub fn create_bind_group_layout(
        &self,
        #[cfg(feature = "debug_labels")] label: Label,
        entries: &[BindGroupLayoutEntry],
    ) -> BindGroupLayout {
        self.device
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
        self.device.create_bind_group(&BindGroupDescriptor {
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
        self.queue
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
        };
        self.device.create_texture(&texture_desc)
    }
}

#[derive(Default)]
pub struct PipelineBuilder<'a> {
    layout: Option<&'a PipelineLayout>,
    vertex_shader: Option<VertexShaderState<'a>>,
    fragment_shader: Option<FragmentShaderState<'a>>,
    primitive: Option<PrimitiveState>,
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
        self.primitive = Some(primitive);
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
    pub fn build(self, state: &State) -> RenderPipeline {
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
    fn to_modules(self, state: &'a State) -> ShaderModules {
        match self {
            ShaderModuleSources::Single(src) => ShaderModules::Single(src.to_module(state)),
            ShaderModuleSources::Multi(vertex_src, fragment_src) => {
                ShaderModules::Multi(vertex_src.to_module(state), fragment_src.to_module(state))
            }
        }
    }
}

pub enum ModuleSrc<'a> {
    Source(ShaderSource<'a>, #[cfg(feature = "debug_labels")] Label<'a>),
    Ref(&'a ShaderModule),
}

impl<'a> ModuleSrc<'a> {
    #[cfg(feature = "debug_labels")]
    fn to_module(self, state: &'a State) -> MaybeOwnedModule<'a> {
        match self {
            ModuleSrc::Source(src, label) => {
                MaybeOwnedModule::Owned(state.create_shader(src, label))
            }
            ModuleSrc::Ref(reference) => MaybeOwnedModule::Ref(reference),
        }
    }

    #[cfg(not(feature = "debug_labels"))]
    fn to_module(self, state: &'a State) -> MaybeOwnedModule<'a> {
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
            MaybeOwnedModule::Ref(reference) => *reference,
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

pub struct TextureBuilder<'a> {
    data: Option<&'a [u8]>,
    dimensions: Option<(u32, u32)>,
    format: Option<TextureFormat>,
    texture_dimension: Option<TextureDimension>,
    usages: TextureUsages,      // MAYBE(currently): we have a default
    aspect: TextureAspect,      // we have a default
    sample_count: u32,          // we have a default
    mip_info: MipInfo,          // we have a default
    depth_or_array_layers: u32, // we have a default
    #[cfg(feature = "debug_labels")]
    label: Label<'a>,
}

impl Default for TextureBuilder<'_> {
    fn default() -> Self {
        Self {
            data: None,
            dimensions: None,
            format: None,
            texture_dimension: None,
            usages: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            aspect: TextureAspect::All,
            sample_count: 1,
            mip_info: MipInfo::default(),
            depth_or_array_layers: 1,
            #[cfg(feature = "debug_labels")]
            label: None,
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

    /// The default value is TextureUsages::TEXTURE_BINDING
    /// *Note*: TextureUsages::COPY_DST gets appended to the usages
    pub fn usages(mut self, usages: TextureUsages) -> Self {
        self.usages = usages | TextureUsages::COPY_DST;
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
        self.sample_count = sample_count;
        self
    }

    /// The default value is `MipInfo::default()`
    #[inline]
    pub fn mip_info(mut self, mip_info: MipInfo) -> Self {
        self.mip_info = mip_info;
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
    pub fn build(self, state: &State) -> Texture {
        state.create_texture(self)
    }
}

pub struct StateBuilder<T: WindowSize> {
    window: Option<T>,
    power_pref: PowerPreference,      // we have a default
    present_mode: PresentMode,        // we have a default
    requirements: DeviceRequirements, // we have a default
    backends: Backends,               // we have a default
    format: Option<TextureFormat>,    // we have a default
}

impl<T: WindowSize> Default for StateBuilder<T> {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            window: None,
            power_pref: Default::default(),
            present_mode: Default::default(),
            requirements: Default::default(),
            format: None,
        }
    }
}

impl<T: WindowSize> StateBuilder<T> {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn window(mut self, window: T) -> Self {
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
pub trait WindowSize: HasRawWindowHandle {
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
