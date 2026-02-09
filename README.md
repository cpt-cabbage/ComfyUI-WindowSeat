# ComfyUI-WindowSeat

A ComfyUI custom node for removing reflections from images using the **WindowSeat** method.

This is an implementation of [**"Reflection Removal through Efficient Adaptation of Diffusion Transformers"**](https://arxiv.org/abs/2512.05000) by Daniyar Zakarin\*, Thiemo Wandel\*, Anton Obukhov, and Dengxin Dai (ETH Zurich / Huawei Bayer Lab). See the [original repository](https://github.com/huawei-bayerlab/windowseat-reflection-removal) for more details on the paper and method.

![WindowSeat Teaser](https://github.com/huawei-bayerlab/windowseat-reflection-removal/raw/main/doc/images/windowseat_teaser.jpg)

![Visualizations](https://github.com/huawei-bayerlab/windowseat-reflection-removal/raw/main/doc/images/visualizations.png)

## Features

- Single-step diffusion inference for fast reflection removal
- Tiled processing with triangular window blending for high-resolution images
- 4-bit NF4 quantization to reduce VRAM usage
- Batch image support
- Adjustable strength, tile size, and other parameters
- Progress bar integration in ComfyUI

## Installation

Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cpt-cabbage/ComfyUI-WindowSeat.git
```

Install the dependencies:

```bash
pip install -r ComfyUI-WindowSeat/requirements.txt
```

For ComfyUI portable (Windows), use the embedded Python instead:

```batch
.\python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WindowSeat\requirements.txt
```

Models are downloaded automatically from Hugging Face on first use.

## Nodes

### Load WindowSeat Model

Downloads and loads the required models:

- **VAE** from [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- **Transformer** (4-bit NF4 quantized) from the same base model
- **LoRA adapter** from [huawei-bayerlab/windowseat-reflection-removal-v1-0](https://huggingface.co/huawei-bayerlab/windowseat-reflection-removal-v1-0)
- **Pre-computed text embeddings** for reflection removal guidance

### WindowSeat Reflection Removal

Processes images through the reflection removal pipeline.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `tile_size` | 768 | 256 - 2048 | Size of square tiles. 768 matches the model training resolution. |
| `max_tiles` | 4 | 1 - 16 | Maximum tiles per axis. More tiles = better quality on large images, but slower. |
| `min_overlap` | 64 | 16 - 256 | Minimum overlap between adjacent tiles in pixels. |
| `batch_size` | 1 | 1 - 4 | Tiles processed simultaneously. Higher = faster but more VRAM. |
| `more_tiles` | False | - | When enabled, uses `tile_size` for tiling. When disabled, uses image short edge (fewer tiles, faster). |
| `strength` | 1.0 | 0.1 - 3.0 | Reflection removal intensity. >1.0 for stubborn reflections, <1.0 for subtler effect. |
| `timestep` | 499 | 1 - 999 | Denoising timestep. The model was trained at 499. |

## Usage

1. Add a **Load WindowSeat Model** node
2. Add a **WindowSeat Reflection Removal** node
3. Connect the model output to the reflection removal node
4. Connect your input image
5. Queue the prompt

The default parameters work well for most images. Increase `strength` if reflections persist, or decrease it for a lighter touch.

## Requirements

- CUDA GPU
- ComfyUI with `diffusers`, `peft`, `bitsandbytes`, `safetensors`, and `torchvision`

## Credits

- **Paper:** [Reflection Removal through Efficient Adaptation of Diffusion Transformers](https://arxiv.org/abs/2512.05000)
- **Authors:** Daniyar Zakarin, Thiemo Wandel, Anton Obukhov, Dengxin Dai
- **Original implementation:** [huawei-bayerlab/windowseat-reflection-removal](https://github.com/huawei-bayerlab/windowseat-reflection-removal)
- **Base model:** [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

## License

Apache License 2.0 — see the [original repository](https://github.com/huawei-bayerlab/windowseat-reflection-removal) for details.
