"""
Dual-tracking control inference for Wan2.2-TI2V-5B.
Supports 2-GPU inference: transformer on cuda:0, VAE+TextEncoder on cuda:1.

Usage:
  python examples/wan2.2_fun/predict_dual_control.py \
    --base_model_path /path/to/Wan2.2-TI2V-5B \
    --tracking_video_1 track1.mp4 --tracking_video_2 track2.mp4 \
    --start_image first_frame.png --prompt "A person riding ..."
"""
import argparse
import gc
import math
import os
import sys

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from videox_fun.models import AutoencoderKLWan3_8, Wan2_2Transformer3DModel, WanT5EncoderModel
from videox_fun.utils.utils import get_video_to_video_latent, get_image_to_video_latent, save_videos_grid


def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self", "cls"}
    return {k: v for k, v in kwargs.items() if k in valid_params}


def main():
    parser = argparse.ArgumentParser(description="Dual-tracking control video generation")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B (https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)")
    parser.add_argument("--config_path", type=str, default="config/wan2.2/wan_dual_control_5b.yaml")
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Trained checkpoint path. If None, uses base_model_path (zero-init test)")
    parser.add_argument("--tracking_video_1", type=str, required=True)
    parser.add_argument("--tracking_video_2", type=str, required=True)
    parser.add_argument("--start_image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_path", type=str, default="output_dual_control.mp4")
    parser.add_argument("--video_length", type=int, default=121)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    weight_dtype = torch.bfloat16

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        dev_dit = torch.device("cuda:0")
        dev_aux = torch.device("cuda:1")
        print(f"2-GPU mode: transformer on cuda:0, VAE+T5 on cuda:1")
    else:
        dev_dit = torch.device("cuda:0")
        dev_aux = torch.device("cuda:0")
        print(f"1-GPU mode: all on cuda:0 (sequential offload)")

    # ── 1. Text Encoder: encode prompt + negative_prompt, then free ──
    print("Loading text encoder...")
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.base_model_path, config.text_encoder_kwargs.get("tokenizer_subpath", "tokenizer")),
    )
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(args.base_model_path, config.text_encoder_kwargs.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=OmegaConf.to_container(config.text_encoder_kwargs),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval().to(dev_aux)

    max_len = config.text_encoder_kwargs.get("text_length", 512)

    def encode_text(text):
        ids = tokenizer([text], padding="max_length", max_length=max_len,
                        truncation=True, add_special_tokens=True, return_tensors="pt")
        input_ids = ids.input_ids.to(dev_aux)
        attn_mask = ids.attention_mask.to(dev_aux)
        seq_len = attn_mask.gt(0).sum(dim=1).long()
        emb = text_encoder(input_ids, attention_mask=attn_mask)[0]
        return [u[:v] for u, v in zip(emb, seq_len)]

    print("  Encoding prompts...")
    prompt_embeds = encode_text(args.prompt)
    negative_prompt_embeds = encode_text(args.negative_prompt)
    prompt_embeds = [e.to(dev_dit) for e in prompt_embeds]
    negative_prompt_embeds = [e.to(dev_dit) for e in negative_prompt_embeds]

    del text_encoder
    gc.collect(); torch.cuda.empty_cache()
    print("  Text encoder freed.")

    # ── 2. VAE: encode tracking videos + first frame ──
    print("Loading VAE...")
    vae = AutoencoderKLWan3_8.from_pretrained(
        os.path.join(args.base_model_path, config.vae_kwargs.get("vae_subpath", "vae")),
        additional_kwargs=OmegaConf.to_container(config.vae_kwargs),
    ).eval().to(dev_aux, dtype=weight_dtype)

    temporal_ratio = vae.config.temporal_compression_ratio
    video_length = int((args.video_length - 1) // temporal_ratio * temporal_ratio) + 1

    print(f"  Encoding tracking video 1...")
    t1, _, _, _ = get_video_to_video_latent(args.tracking_video_1, video_length=video_length,
                                             sample_size=[args.height, args.width])
    t1 = t1.to(dev_aux, dtype=weight_dtype) * 2.0 - 1.0
    with torch.no_grad():
        tracking1_latents = vae.encode(t1)[0].sample()
    del t1

    print(f"  Encoding tracking video 2...")
    t2, _, _, _ = get_video_to_video_latent(args.tracking_video_2, video_length=video_length,
                                             sample_size=[args.height, args.width])
    t2 = t2.to(dev_aux, dtype=weight_dtype) * 2.0 - 1.0
    with torch.no_grad():
        tracking2_latents = vae.encode(t2)[0].sample()
    del t2

    control_latents = torch.cat([tracking1_latents, tracking2_latents], dim=1)
    del tracking1_latents, tracking2_latents
    print(f"  Control latents: {control_latents.shape}")

    # Encode ONLY 1 frame (aligned with official WanTI2V.i2v). In blending,
    # ti2v_latents broadcasts along the T dim; mask makes only frame 0 used.
    print(f"  Encoding first frame (single-frame, broadcast in blending)...")
    start_img, _, _ = get_image_to_video_latent(args.start_image, None, video_length=1,
                                                 sample_size=[args.height, args.width])
    start_img = start_img.to(dev_aux, dtype=weight_dtype) * 2.0 - 1.0
    with torch.no_grad():
        ti2v_latents = vae.encode(start_img)[0].sample()   # [1, 48, 1, H_lat, W_lat]
    del start_img

    # Full latent shape (all frames) — taken from control_latents (same T/H/W).
    _, _, F_lat_full, H_lat, W_lat = control_latents.shape
    latent_shape = (1, ti2v_latents.shape[1], F_lat_full, H_lat, W_lat)
    print(f"  ti2v_latents: {tuple(ti2v_latents.shape)}, full latent shape: {latent_shape}")

    ti2v_mask = torch.ones(1, 1, F_lat_full, H_lat, W_lat, device=dev_dit, dtype=weight_dtype)
    ti2v_mask[:, :, :1, :, :] = 0

    ti2v_latents = ti2v_latents.to(dev_dit)
    control_latents = control_latents.to(dev_dit)

    vae.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    # ── 3. Load Transformer ──
    print("Loading transformer...")
    if args.transformer_path:
        transformer_full_path = os.path.join(args.transformer_path, "transformer")
    else:
        sub_path = config.transformer_additional_kwargs.get("transformer_low_noise_model_subpath", "transformer")
        transformer_full_path = os.path.join(args.base_model_path, sub_path)
    transformer = Wan2_2Transformer3DModel.from_pretrained(
        transformer_full_path,
        transformer_additional_kwargs=OmegaConf.to_container(config.transformer_additional_kwargs),
    ).eval().to(dev_dit, dtype=weight_dtype)

    pe = transformer.patch_embedding.weight
    print(f"  patch_embedding: {pe.shape}")
    print(f"    [:, :48] norm = {pe[:, :48].float().norm():.2f} (pretrained)")
    if pe.shape[1] > 48:
        print(f"    [:, 48:] norm = {pe[:, 48:].float().norm():.6f} (tracking, expect ~0 for zero-init)")

    # ── 4. Compute seq_len ──
    F_lat, H_lat, W_lat = latent_shape[2], latent_shape[3], latent_shape[4]
    ps_h, ps_w = transformer.config.patch_size[1], transformer.config.patch_size[2]
    seq_len = math.ceil(W_lat * H_lat / (ps_h * ps_w) * F_lat)
    print(f"  seq_len = {seq_len}")

    # ── 5. Init noise ──
    generator = torch.Generator(device=dev_dit).manual_seed(args.seed)
    latents = torch.randn(latent_shape, device=dev_dit, generator=generator, dtype=weight_dtype)

    # ── 6. Scheduler ──
    scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config.scheduler_kwargs))
    )
    scheduler.set_timesteps(args.num_inference_steps, device=dev_dit)
    timesteps = scheduler.timesteps

    # ── 7. Denoising loop ──
    do_cfg = args.guidance_scale > 1.0
    print(f"Denoising: {args.num_inference_steps} steps, guidance_scale={args.guidance_scale}, CFG={'on' if do_cfg else 'off'}")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=weight_dtype):
        for i, t in enumerate(timesteps):
            latent_model_input = (1 - ti2v_mask) * ti2v_latents + ti2v_mask * latents

            temp_ts = (ti2v_mask[:, 0, :, ::ps_h, ::ps_w] * t).flatten(1)
            timestep = torch.cat([temp_ts, temp_ts.new_ones(1, seq_len - temp_ts.size(1)) * t], dim=1)

            noise_cond = transformer(x=latent_model_input, context=prompt_embeds,
                                     t=timestep, seq_len=seq_len, y=control_latents)
            if do_cfg:
                noise_uncond = transformer(x=latent_model_input, context=negative_prompt_embeds,
                                           t=timestep, seq_len=seq_len, y=control_latents)
                noise_pred = noise_uncond + args.guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = noise_cond

            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = (1 - ti2v_mask) * ti2v_latents + ti2v_mask * latents

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  step {i+1}/{args.num_inference_steps}")

    print("Denoising done.")
    del transformer
    gc.collect(); torch.cuda.empty_cache()

    # ── 8. VAE decode ──
    print("VAE decoding...")
    vae.to(dev_aux)
    with torch.no_grad():
        video = vae.decode(latents.to(dev_aux, dtype=weight_dtype)).sample
    video = (video / 2 + 0.5).clamp(0, 1).cpu().float()

    del vae, latents
    gc.collect(); torch.cuda.empty_cache()

    # ── 9. Save ──
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    save_videos_grid(video, os.path.abspath(args.output_path), fps=args.fps)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
