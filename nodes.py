import gc
import json

import safetensors.torch
import torch
from diffusers import (
    AutoencoderKLQwenImage,
    BitsAndBytesConfig,
    QwenImageTransformer2DModel,
)
from huggingface_hub import hf_hub_download
from peft import LoraConfig

import comfy.utils
import model_management

from . import pipeline

BASE_MODEL_URI = "Qwen/Qwen-Image-Edit-2509"
LORA_MODEL_URI = "huawei-bayerlab/windowseat-reflection-removal-v1-0"


class WindowSeatModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
        }

    RETURN_TYPES = ("WINDOWSEAT_MODEL",)
    RETURN_NAMES = ("windowseat_model",)
    FUNCTION = "load_model"
    CATEGORY = "WindowSeat"

    def load_model(self):
        device = model_management.get_torch_device()

        # Read processing resolution from model config
        config_file = hf_hub_download(LORA_MODEL_URI, "model_index.json")
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        processing_resolution = config_dict["processing_resolution"]

        print("[WindowSeat] Loading VAE...")
        vae = AutoencoderKLQwenImage.from_pretrained(
            BASE_MODEL_URI,
            subfolder="vae",
            torch_dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        vae.to(device, dtype=torch.bfloat16)

        print("[WindowSeat] Loading Transformer (4-bit NF4 quantized)...")
        nf4 = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["transformer_blocks.0.img_mod"],
        )
        transformer = QwenImageTransformer2DModel.from_pretrained(
            BASE_MODEL_URI,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            quantization_config=nf4,
            device_map=device,
        )

        print("[WindowSeat] Applying LoRA adapter...")
        lora_config = LoraConfig.from_pretrained(LORA_MODEL_URI, subfolder="transformer_lora")
        transformer.add_adapter(lora_config)
        lora_weights_path = hf_hub_download(
            LORA_MODEL_URI, "pytorch_lora_weights.safetensors", subfolder="transformer_lora"
        )
        state_dict = safetensors.torch.load_file(lora_weights_path)
        missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected keys in LoRA state dict: {unexpected}")

        print("[WindowSeat] Loading text embeddings...")
        embeds_path = hf_hub_download(
            LORA_MODEL_URI, "state_dict.safetensors", subfolder="text_embeddings"
        )
        embeds_dict = safetensors.torch.load_file(embeds_path)

        print("[WindowSeat] Model loaded successfully.")

        model_dict = {
            "vae": vae,
            "transformer": transformer,
            "embeds_dict": embeds_dict,
            "processing_resolution": processing_resolution,
            "device": device,
        }

        return (model_dict,)


class WindowSeatReflectionRemoval:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "windowseat_model": ("WINDOWSEAT_MODEL",),
                "image": ("IMAGE",),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 768, "min": 256, "max": 2048, "step": 64,
                    "tooltip": "Size of square tiles for processing. Default 768 matches the model training resolution.",
                }),
                "max_tiles": ("INT", {
                    "default": 4, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Maximum number of tiles per axis. More tiles = higher quality on large images but slower.",
                }),
                "min_overlap": ("INT", {
                    "default": 64, "min": 16, "max": 256, "step": 16,
                    "tooltip": "Minimum overlap between adjacent tiles in pixels.",
                }),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Number of tiles to process simultaneously. Higher = faster but uses more VRAM.",
                }),
                "more_tiles": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, uses tile_size for tiling. When disabled, uses the image short edge (fewer tiles, faster).",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05,
                    "tooltip": "Reflection removal intensity. 1.0 = default. Values above 1.0 push harder to remove stubborn reflections. Values below 1.0 give a subtler effect.",
                }),
                "timestep": ("INT", {
                    "default": 499, "min": 1, "max": 999, "step": 1,
                    "tooltip": "Advanced: denoising timestep. The model was trained at 499. Higher values may predict stronger corrections but risk artifacts.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "remove_reflections"
    CATEGORY = "WindowSeat"

    def remove_reflections(self, windowseat_model, image, tile_size=768, max_tiles=4, min_overlap=64, batch_size=1, more_tiles=False, strength=1.0, timestep=499):
        vae = windowseat_model["vae"]
        transformer = windowseat_model["transformer"]
        embeds_dict = windowseat_model["embeds_dict"]
        processing_resolution = windowseat_model["processing_resolution"]

        # ComfyUI IMAGE: [N, H, W, C] float32 [0,1] -> [N, C, H, W] float32 [-1,1]
        img_batch = image.permute(0, 3, 1, 2) * 2.0 - 1.0
        N = img_batch.shape[0]

        # Count total tiles for progress bar
        total_tiles = 0
        for i in range(N):
            _, _, H, W = img_batch[i:i+1].shape
            tiles, _, _ = pipeline.compute_tiles(
                W, H, tile_size, max_tiles, max_tiles,
                min_overlap, min_overlap,
                not more_tiles, processing_resolution,
            )
            total_tiles += len(tiles)

        pbar = comfy.utils.ProgressBar(total_tiles)
        tiles_done = [0]

        def progress_callback(completed):
            delta = completed - (tiles_done[0] % 10000)
            tiles_done[0] += delta
            pbar.update(delta)

        results = []
        for i in range(N):
            single_img = img_batch[i]  # [C, H, W]

            result_chw = pipeline.process_single_image(
                single_img,
                vae,
                transformer,
                embeds_dict,
                processing_resolution,
                tile_size=tile_size,
                max_tiles=max_tiles,
                min_overlap=min_overlap,
                batch_size=batch_size,
                more_tiles=more_tiles,
                strength=strength,
                timestep=timestep,
                progress_callback=progress_callback,
            )

            # [C, H, W] [0,1] -> [H, W, C]
            result_hwc = result_chw.permute(1, 2, 0).clamp(0.0, 1.0)
            results.append(result_hwc)

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        output = torch.stack(results, dim=0)  # [N, H, W, C]
        return (output,)


NODE_CLASS_MAPPINGS = {
    "WindowSeatModelLoader": WindowSeatModelLoader,
    "WindowSeatReflectionRemoval": WindowSeatReflectionRemoval,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WindowSeatModelLoader": "Load WindowSeat Model",
    "WindowSeatReflectionRemoval": "WindowSeat Reflection Removal",
}
