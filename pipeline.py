import math

import numpy as np
import torch
import torchvision
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageEditPipeline,
    QwenImageTransformer2DModel,
)
from PIL import Image


def _required_side_for_axis(size: int, nmax: int, min_overlap: int) -> int:
    nmax = max(1, int(nmax))
    if nmax == 1:
        return size
    return math.ceil((size + (nmax - 1) * min_overlap) / nmax)


def _starts(size: int, T: int, min_overlap: int):
    if size <= T:
        return [0]
    stride = max(1, T - min_overlap)
    xs = list(range(0, size - T + 1, stride))
    last = size - T
    if xs[-1] != last:
        xs.append(last)
    out = []
    for v in xs:
        if not out or v > out[-1]:
            out.append(v)
    return out


def compute_tiles(W, H, tile_size, max_tiles_w, max_tiles_h, min_overlap_w, min_overlap_h, use_short_edge_tile, processing_resolution):
    if use_short_edge_tile:
        short_edge = max(min(W, H), processing_resolution)
        pref_side = short_edge
    else:
        pref_side = tile_size

    actual_W, actual_H = W, H
    if W < pref_side or H < pref_side:
        min_side = min(W, H)
        scale_ratio = pref_side / min_side
        actual_W = round(scale_ratio * W)
        actual_H = round(scale_ratio * H)

    Nw, Nh = int(max_tiles_w), int(max_tiles_h)
    ow, oh = int(min_overlap_w), int(min_overlap_h)

    T_low = max(
        _required_side_for_axis(actual_W, Nw, ow),
        _required_side_for_axis(actual_H, Nh, oh),
        ow + 1,
        oh + 1,
    )
    T_high = min(actual_W, actual_H)

    if T_low > T_high:
        raise ValueError(
            f"Infeasible tiling constraints: need T >= {T_low}, but max square is {T_high}. "
            f"Try increasing max_tiles or reducing tile_size."
        )

    T = max(T_low, min(pref_side, T_high))

    xs = _starts(actual_W, T, ow)
    ys = _starts(actual_H, T, oh)

    tiles = [(x, y, x + T, y + T) for y in ys for x in xs]
    return tiles, actual_W, actual_H


def lanczos_resize_chw(x, out_hw):
    H_out, W_out = map(int, out_hw)

    is_torch = isinstance(x, torch.Tensor)
    if is_torch:
        dev = x.device
        arr = x.detach().cpu().numpy()
    else:
        arr = x

    chw = arr.astype(np.float32, copy=False)
    C = chw.shape[0]

    out_chw = np.empty((C, H_out, W_out), dtype=np.float32)
    for c in range(C):
        ch = chw[c]
        img = Image.fromarray(ch).convert("F")
        img = img.resize((W_out, H_out), resample=Image.LANCZOS)
        out_chw[c] = np.asarray(img, dtype=np.float32)

    if is_torch:
        return torch.from_numpy(out_chw).to(dev)
    return out_chw


def encode(image: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    image = image.to(device=vae.device, dtype=vae.dtype)
    out = vae.encode(image.unsqueeze(2)).latent_dist.sample()
    latents_mean = torch.tensor(vae.config.latents_mean, device=out.device, dtype=out.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=out.device, dtype=out.dtype)
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    out = (out - latents_mean) * latents_std_inv
    return out


def decode(latents: torch.Tensor, vae: AutoencoderKLQwenImage) -> torch.Tensor:
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
    latents_mean = latents_mean.view(1, vae.config.z_dim, 1, 1, 1)
    latents_std_inv = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
    latents_std_inv = latents_std_inv.view(1, vae.config.z_dim, 1, 1, 1)
    latents = latents / latents_std_inv + latents_mean
    out = vae.decode(latents)
    out = out.sample[:, :, 0]
    return out


def _match_batch(t: torch.Tensor, B: int) -> torch.Tensor:
    if t.size(0) == B:
        return t
    if t.size(0) == 1 and B > 1:
        return t.expand(B, *t.shape[1:])
    if t.size(0) > B:
        return t[:B]
    reps = (B + t.size(0) - 1) // t.size(0)
    return t.repeat((reps,) + (1,) * (t.ndim - 1))[:B]


def flow_step(
    model_input: torch.Tensor,
    transformer: QwenImageTransformer2DModel,
    vae: AutoencoderKLQwenImage,
    embeds_dict: dict[str, torch.Tensor],
    timestep: int = 499,
) -> torch.Tensor:
    prompt_embeds = embeds_dict["prompt_embeds"]
    prompt_mask = embeds_dict["prompt_mask"]

    if prompt_mask.dtype != torch.bool:
        prompt_mask = prompt_mask > 0

    if model_input.ndim == 5 and model_input.shape[2] == 1:
        model_input_4d = model_input[:, :, 0]
    elif model_input.ndim == 4:
        model_input_4d = model_input
    else:
        raise ValueError(f"Unexpected model_input shape: {model_input.shape}")

    B, C, H, W = model_input_4d.shape
    device = next(transformer.parameters()).device

    prompt_embeds = _match_batch(prompt_embeds, B).to(device=device, dtype=torch.bfloat16, non_blocking=True)
    prompt_mask = _match_batch(prompt_mask, B).to(device=device, dtype=torch.bool, non_blocking=True)

    packed_model_input = QwenImageEditPipeline._pack_latents(
        model_input_4d,
        batch_size=B,
        num_channels_latents=C,
        height=H,
        width=W,
    )
    packed_model_input = packed_model_input.to(torch.bfloat16)

    timestep = torch.full((B,), float(timestep), device=device, dtype=torch.bfloat16) / 1000.0

    h_img = H // 2
    w_img = W // 2
    img_shapes = [[(1, h_img, w_img)]] * B
    txt_seq_lens = prompt_mask.sum(dim=1).tolist()

    attention_kwargs = getattr(transformer, "attention_kwargs", None) or {}

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=packed_model_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            guidance=None,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]

    temperal_downsample = vae.config.get("temperal_downsample", None)
    if temperal_downsample is not None:
        vae_scale_factor = 2 ** len(temperal_downsample)
    else:
        vae_scale_factor = 8

    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred,
        height=H * vae_scale_factor,
        width=W * vae_scale_factor,
        vae_scale_factor=vae_scale_factor,
    )

    latent_output = model_input.to(vae.dtype) - model_pred.to(vae.dtype)
    return latent_output


@torch.no_grad()
def process_single_image(
    img_chw_norm,
    vae,
    transformer,
    embeds_dict,
    processing_resolution,
    tile_size=768,
    max_tiles=4,
    min_overlap=64,
    batch_size=1,
    more_tiles=False,
    strength=1.0,
    timestep=499,
    progress_callback=None,
):
    """Process a single image through the WindowSeat reflection removal pipeline.

    Args:
        img_chw_norm: [C, H, W] float32 tensor in [-1, 1]
        progress_callback: callable(num_completed_tiles) for progress updates

    Returns:
        [C, H, W] float32 tensor in [0, 1]
    """
    C, orig_H, orig_W = img_chw_norm.shape
    use_short_edge_tile = not more_tiles

    tiles, working_W, working_H = compute_tiles(
        orig_W, orig_H, tile_size,
        max_tiles, max_tiles,
        min_overlap, min_overlap,
        use_short_edge_tile, processing_resolution,
    )

    if working_W != orig_W or working_H != orig_H:
        working_img = lanczos_resize_chw(img_chw_norm, (working_H, working_W))
    else:
        working_img = img_chw_norm

    tile_predictions = []
    completed = 0

    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start:batch_start + batch_size]
        batch_inputs = []

        for (x0, y0, x1, y1) in batch_tiles:
            tile_crop = working_img[:, y0:y1, x0:x1]
            tile_resized = lanczos_resize_chw(tile_crop, (processing_resolution, processing_resolution))
            if isinstance(tile_resized, np.ndarray):
                tile_resized = torch.from_numpy(tile_resized)
            batch_inputs.append(tile_resized)

        batch_tensor = torch.stack(batch_inputs, dim=0)

        latents = encode(batch_tensor, vae)
        latents = flow_step(latents, transformer, vae, embeds_dict, timestep=timestep)
        pixel_pred = decode(latents, vae)

        for j, tile_info in enumerate(batch_tiles):
            tile_predictions.append({
                "pred": pixel_pred[j].cpu().float(),
                "tile_info": tile_info,
            })

        completed += len(batch_tiles)
        if progress_callback:
            progress_callback(completed)

    # Stitch tiles with triangular window blending
    acc = torch.zeros(3, working_H, working_W, dtype=torch.float32)
    wsum = torch.zeros(working_H, working_W, dtype=torch.float32)

    for t in tile_predictions:
        x0, y0, x1, y1 = t["tile_info"]
        tile = t["pred"].float()
        if tile.ndim == 4:
            tile = tile.squeeze(0)

        tH, tW = y1 - y0, x1 - x0
        h, w = tile.shape[-2:]
        if h != tH or w != tW:
            tile = lanczos_resize_chw(tile, (tH, tW))
            if isinstance(tile, np.ndarray):
                tile = torch.from_numpy(tile)
            h, w = tH, tW

        wx = 1 - (2 * torch.arange(w, dtype=torch.float32) / max(w - 1, 1) - 1).abs()
        wy = 1 - (2 * torch.arange(h, dtype=torch.float32) / max(h - 1, 1) - 1).abs()
        w2 = (wy[:, None] * wx[None, :]).clamp_min(1e-3)

        acc[:, y0:y1, x0:x1] += tile * w2
        wsum[y0:y1, x0:x1] += w2

    stitched = acc / wsum.clamp_min(1e-6)

    # Convert from [-1,1] to [0,1] and resize to original dimensions
    stitched_01 = ((stitched + 1.0) / 2.0).clamp(0.0, 1.0)

    if working_H != orig_H or working_W != orig_W:
        pil = torchvision.transforms.functional.to_pil_image(stitched_01)
        pil_resized = pil.resize((orig_W, orig_H), resample=Image.LANCZOS)
        result = torchvision.transforms.functional.to_tensor(pil_resized)
    else:
        result = stitched_01

    # Apply strength as pixel-space blend between original and processed
    if strength != 1.0:
        original_01 = (img_chw_norm + 1.0) / 2.0
        result = original_01 * (1.0 - strength) + result * strength
        result = result.clamp(0.0, 1.0)

    return result
