use std::{
    collections::BTreeMap,
    io::{Read, Seek, SeekFrom},
    mem::ManuallyDrop,
    sync::Arc,
};

use anyhow::{Context, Result, ensure};
use glam::{Vec2, vec2};
use shin_core::format::picture::{
    PicBlock, PicBlockRect, PictureBuilder, PictureFormat, read_picture_header_prefix_with_format,
};

use shin_render::{
    gpu_texture::GpuTexture,
    shaders::types::{
        buffer::{Buffer, OwnedIndexBuffer, OwnedVertexBuffer},
        vertices::PosTexVertex,
    },
};

use crate::asset::system::{Asset, AssetDataAccessor, AssetLoadContext};

#[cfg(test)]
use shin_core::format::picture::read_picture_header_prefix;
//use std::fs::File;

pub struct GpuPictureBlock {
    pub native_format: u16,
    /// Original block+8/+a/+c/+e: signed origins, unsigned texture extents.
    pub native_bounds: [u16; 4],
    pub vertex_buffer: ManuallyDrop<OwnedVertexBuffer<PosTexVertex>>,
    pub index_buffer: ManuallyDrop<OwnedIndexBuffer>,
    pub opaque_rect_count: usize,
    pub transparent_rect_count: usize,
    pub texture: ManuallyDrop<GpuTexture>,
}

pub struct ExtPictureHeader {
    //pub picture_id: u32,
    pub origin_x: i16,
    pub origin_y: i16,
}
impl ExtPictureHeader {
    pub fn new() -> Self {
        return ExtPictureHeader{
           // picture_id: 0,
            origin_x: 0,
            origin_y: 0,
        }
    }
    /*
    pub fn is_unset(&mut self) -> bool {
        return self.origin_x == 0 && self.origin_y == 0 && self.picture_id == 0;
    }
    */
}

pub fn parse_ext_picture_header(bytes: &[u8]) -> ExtPictureHeader {
    if bytes.len() < 64 {
        println!("Bad input {} ({}) !", String::from_utf8_lossy(bytes), bytes.len());
        return ExtPictureHeader::new();
    }
    let start = bytes.len() - 64;
    if start == 0 {
        println!("Failed to parse PIC header");
        return ExtPictureHeader::new();
    }
    if &bytes[start..start+7] != b"[PIC_B]" {
        println!("Unable to find PIC header end");
        return ExtPictureHeader::new();
    }

    let mut proto_header = &bytes[start..(bytes.len())];
    let eof = String::from_utf8_lossy(proto_header).find("[PIC_E]").expect("Failed to parse proto_header");
    proto_header = &proto_header[7..eof];
    let header_text = String::from_utf8_lossy(proto_header);
    let header_values: Vec<&str> = header_text.split("|").collect();
    let origin_res: Vec<&str> = header_values[1].split("x").collect();
    if origin_res.len() != 2 {
        println!("[debug] header_text {}, origin_res {:?} ({})", header_text, origin_res, origin_res.len());
        println!("Unable to parse origin_x and origin_y from header!")
    }
    /*
    let offset_res: Vec<&str> = header_values[2].split("x").collect();
    if offset_res.len() != 2 {
        println!("Unable to parse offset_x and offset_y from hreader!")
    }
    */
    return ExtPictureHeader{
        //picture_id: header_values[0].parse().expect("Unable to parse picture_id"),
        origin_x: origin_res[0].to_string().parse().expect("Unable to parse origin_x"),
        origin_y: origin_res[1].to_string().parse().expect("Unable to parse origin_y"),
        //offset_x: offset_res[0].to_string().parse().expect("Unable to parse offset_x"),
        //offset_y: offset_res[1].to_string().parse().expect("Unable to parse offset_y"),
    }
}

impl Drop for GpuPictureBlock {
    /// Releasing a block's Metal texture and buffers takes milliseconds, and a
    /// scene change (most visibly while skipping) drops many blocks at once,
    /// so release them on a compute thread instead of the caller's.
    fn drop(&mut self) {
        // SAFETY: each field is taken exactly once and never used again.
        let resources = unsafe {
            (
                ManuallyDrop::take(&mut self.vertex_buffer),
                ManuallyDrop::take(&mut self.index_buffer),
                ManuallyDrop::take(&mut self.texture),
            )
        };
        shin_tasks::compute::spawn_and_forget(move || drop(resources));
    }
}

impl GpuPictureBlock {
    pub const VERTICES_PER_RECT: usize = 4;
    pub const INDICES_PER_RECT: usize = 6;

    pub fn new(context: GpuTextureBuilderContext, block: PicBlock, label: &str) -> Self {
        let (vertex_buffer, index_buffer) = vertex_data(&block);

        let vertex_buffer = Buffer::allocate_vertex(
            context.wgpu_device,
            &vertex_buffer,
            Some(&format!("{}/vertex", label)),
        );
        let index_buffer = Buffer::allocate_index(
            context.wgpu_device,
            &index_buffer,
            Some(&format!("{}/index", label)),
        );

        let texture = GpuTexture::new_static_from_rgba_image(
            context.wgpu_device,
            context.wgpu_queue,
            Some(label),
            &block.data,
        );

        GpuPictureBlock {
            native_format: block.native_format,
            native_bounds: [
                block.offset_x as u16,
                block.offset_y as u16,
                block.data.width() as u16,
                block.data.height() as u16,
            ],
            vertex_buffer: ManuallyDrop::new(vertex_buffer),
            index_buffer: ManuallyDrop::new(index_buffer),
            opaque_rect_count: block.opaque_rects.len(),
            transparent_rect_count: block.transparent_rects.len(),
            texture: ManuallyDrop::new(texture),
        }
    }
}

pub(super) fn vertex_data(block: &PicBlock) -> (Vec<PosTexVertex>, Vec<u16>) {
    let rect_count = block.opaque_rects.len() + block.transparent_rects.len();

    let mut vertex_buffer = Vec::with_capacity(rect_count * GpuPictureBlock::VERTICES_PER_RECT);
    let mut index_buffer = Vec::with_capacity(rect_count * GpuPictureBlock::INDICES_PER_RECT);

    // Native block geometry sign-extends both 16-bit offsets and computes each
    // reciprocal once. Dividing every UV separately differs by an f32 ULP.
    let offset_x = block.offset_x as u16 as i16 as f32;
    let offset_y = block.offset_y as u16 as i16 as f32;

    let width = block.data.width() as f32;
    let height = block.data.height() as f32;
    let inverse_width = 1.0 / width;
    let inverse_height = 1.0 / height;

    let mut emit_rect = |PicBlockRect {
                             from_x,
                             from_y,
                             to_x,
                             to_y,
                         }: PicBlockRect| {
        let from_x = from_x as f32;
        let to_x = to_x as f32;
        let from_y = from_y as f32;
        let to_y = to_y as f32;

        let x_left = (offset_x + from_x, inverse_width * (from_x + 1.0));
        let x_right = (
            (to_x + 0.5) + offset_x,
            inverse_width * ((to_x + 0.5) + 1.0),
        );
        let y_top = (offset_y + from_y, inverse_height * (from_y + 1.0));
        let y_bottom = (
            (to_y + 0.5) + offset_y,
            inverse_height * ((to_y + 0.5) + 1.0),
        );

        let vertices: [_; GpuPictureBlock::VERTICES_PER_RECT] = [
            (x_left, y_top),
            (x_right, y_top),
            (x_left, y_bottom),
            (x_right, y_bottom),
        ];

        let index_base = vertex_buffer.len() as u16;
        let indices: [u16; GpuPictureBlock::INDICES_PER_RECT] =
            [0, 1, 2, 3, 2, 1].map(|i| index_base + i);

        vertex_buffer.extend(vertices.map(|((px, tx), (py, ty))| PosTexVertex {
            position: vec2(px, py),
            texture_position: vec2(tx, ty),
        }));
        index_buffer.extend(indices);
    };

    block.opaque_rects.iter().cloned().for_each(&mut emit_rect);
    block
        .transparent_rects
        .iter()
        .cloned()
        .for_each(&mut emit_rect);

    (vertex_buffer, index_buffer)
}

#[cfg(test)]
mod native_load_tests;

#[cfg(test)]
mod cache_tests;

#[cfg(test)]
mod header_tests;

#[derive(Copy, Clone)]
pub struct GpuTextureBuilderContext<'a> {
    pub wgpu_device: &'a wgpu::Device,
    pub wgpu_queue: &'a wgpu::Queue,
}

struct GpuPictureBuilder<'a> {
    context: GpuTextureBuilderContext<'a>,
    label: String,
    blocks: PictureBlocks,
}

impl<'a> PictureBuilder for GpuPictureBuilder<'a> {
    type Args = (GpuTextureBuilderContext<'a>, String);
    type Output = PictureBlocks;

    fn new(
        (context, label): Self::Args,
        _effective_width: u32,
        _effective_height: u32,
        _origin_x: i32,
        _origin_y: i32,
        _picture_id: u32,
    ) -> Self {
        GpuPictureBuilder {
            context,
            label,
            blocks: PictureBlocks {
                blocks: BTreeMap::new(),
                placements: Vec::new(),
            },
        }
    }

    fn add_block(
        &mut self,
        data_offset: u32,
        _positions: Vec<(u32, u32)>,
        block: PicBlock,
    ) -> Result<()> {
        // The native block constructor returns null before decoding/uploading
        // when neither rectangle group has geometry.
        if block.opaque_rects.is_empty() && block.transparent_rects.is_empty() {
            return Ok(());
        }
        let block = GpuPictureBlock::new(
            self.context,
            block,
            &format!("{}/{}", self.label, data_offset),
        );
        assert!(self.blocks.blocks.insert(data_offset, block).is_none());

        Ok(())
    }

    fn set_placements(&mut self, placements: Vec<(u32, (u32, u32))>) {
        self.blocks.placements = placements
            .into_iter()
            .map(|(offset, (x, y))| (offset, vec2(x as f32, y as f32)))
            .collect();
    }

    fn build(self) -> Result<Self::Output> {
        Ok(self.blocks)
    }
}

pub struct PictureBlocks {
    blocks: BTreeMap<u32, GpuPictureBlock>,
    placements: Vec<(u32, Vec2)>,
}

impl PictureBlocks {
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    pub fn iter_placements(&self) -> impl Iterator<Item = (u32, Vec2, &GpuPictureBlock)> {
        self.placements.iter().filter_map(|&(offset, position)| {
            self.blocks
                .get(&offset)
                .map(|block| (offset, position, block))
        })
    }
}

/// Per-file metadata and native-ID shared placement/GPU block ownership.
pub struct Picture {
    pub label: String,
    pub effective_width: u32,
    pub effective_height: u32,
    pub origin_x: i32,
    pub origin_y: i32,
    // Preserve native file metadata for header validation and future flag handling.
    #[allow(dead_code)]
    pub flags: u32,
    pub scale: f32,
    pub blocks: Arc<PictureBlocks>,
}

fn is_webp(data: &[u8]) -> bool {
    data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WEBP"
}

/// WebP replacements have no native header
fn decode_webp_picture(
    context: GpuTextureBuilderContext,
    label: String,
    data: &[u8],
) -> Result<(PictureBlocks, u32, u32, ExtPictureHeader)> {
    let ext_header = parse_ext_picture_header(data);
    let image = image::load_from_memory_with_format(data, image::ImageFormat::WebP)
        .context("Decoding WebP picture")?
        .to_rgba8();
    let (width, height) = image.dimensions();
    ensure!(
        width > 0 && height > 0,
        "WebP picture has zero dimensions"
    );
    ensure!(
        width <= u16::MAX as u32 && height <= u16::MAX as u32,
        "WebP picture dimensions exceed the native picture geometry range"
    );

    let rect = PicBlockRect {
        from_x: 0,
        from_y: 0,
        to_x: (width - 1) as u16,
        to_y: (height - 1) as u16,
    };

    let has_alpha = image.pixels().any(|pixel| pixel[3] != u8::MAX);
    let mut block = PicBlock::new(0, 0, width, height, Vec::new(), Vec::new());
    if has_alpha {
        block.transparent_rects.push(rect);
    } else {
        block.opaque_rects.push(rect);
    }
    block.data = image;

    let mut builder = GpuPictureBuilder::new((context, label), width, height, 0, 0, 0);
    builder.add_block(0, vec![(0, 0)], block)?;
    builder.set_placements(vec![(0, (0, 0))]);
    Ok((builder.build()?, width, height, ext_header))
}

impl Asset for Picture {
    type Args = ();

    async fn load(
        context: &Arc<AssetLoadContext>,
        _args: (),
        name: &str,
        data: AssetDataAccessor,
    ) -> Result<Self> {
        let label = format!("Pic[{}]", name);
        let format = match context.game {
            crate::cli::Game::Umineko | crate::cli::Game::Gerokasu2 | crate::cli::Game::Gerokasu => PictureFormat::Umineko,
            crate::cli::Game::Higurashi => PictureFormat::HigurashiV2,
        };
        let mut cursor = data.cursor()?;
        let size = cursor.seek(SeekFrom::End(0))?;
        cursor.seek(SeekFrom::Start(0))?;
        let mut prefix = vec![
            0;
            if format == PictureFormat::HigurashiV2 {
                32
            } else {
                36
            }
        ];
        cursor.read_exact(&mut prefix)?;

        if is_webp(&prefix) {
            let load_label = label.clone();
            let load_context = context.clone();
            let (blocks, effective_width, effective_height, ext_header) =
                shin_tasks::compute::spawn(move || {
                    crate::asset::system::catch_asset_preparation_panic("Picture", || {
                        let mut data = prefix;
                        cursor.read_to_end(&mut data)?;
                        decode_webp_picture(
                            GpuTextureBuilderContext {
                                wgpu_device: &load_context.wgpu_device,
                                wgpu_queue: &load_context.wgpu_queue,
                            },
                            load_label,
                            &data,
                        )
                    })
                }).await?;
            //let mut origin_x = (effective_width / 2) as i16;
            //let mut origin_y = (effective_height / 2) as i16;
            //let ext_header = parse_ext_picture_header(&data);
            return Ok(Self {
                label,
                effective_width,
                effective_height,
                origin_x:   ext_header.origin_x.into(),
                origin_y:   ext_header.origin_y.into(),
                flags: 1,
                scale: 1.0, // 4096 as f32 * (1.0 / 4096.0)
                blocks: Arc::new(blocks),
            });
        }

        let header = read_picture_header_prefix_with_format(&prefix, size, format)?;
        let load_label = label.clone();
        let load_context = context.clone();
        let blocks = context
            .picture_cache
            .get_or_load(header.picture_id, || async move {
                // Cache hits need only the fixed header, as in the native loader.
                // Decode/upload runs on compute workers without blocking awaiters.
                shin_tasks::compute::spawn(move || {
                    crate::asset::system::catch_asset_preparation_panic("Picture", || {
                        let mut data = prefix.to_vec();
                        cursor.read_to_end(&mut data)?;
                        shin_core::format::picture::read_picture_with_format::<GpuPictureBuilder>(
                            &data,
                            (
                                GpuTextureBuilderContext {
                                    wgpu_device: &load_context.wgpu_device,
                                    wgpu_queue: &load_context.wgpu_queue,
                                },
                                load_label,
                            ),
                            format,
                        )
                    })
                })
                .await
            })
            .await?;
        Ok(Self {
            label,
            effective_width: header.effective_width.into(),
            effective_height: header.effective_height.into(),
            origin_x: header.origin_x.into(),
            origin_y: header.origin_y.into(),
            flags: header.flags,
            scale: header.scale as f32 * (1.0 / 4096.0),
            blocks,
        })
    }
}
