from __future__ import annotations

import os
import sys

sys.path.append(".")
import math
import random
from argparse import ArgumentParser

import einops
from edm_sampler.external import CompVisDenoiser
from edm_sampler.sampling import sample_euler_ancestral
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

from stable_diffusion.ldm.util import instantiate_from_config
from constants.const import theme_available



class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--cfg-text-list",
                        default=[0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                        nargs="+", type=float)
    parser.add_argument("--cfg-image-list",
                        default=[0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
                        nargs="+", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Extract the file name from a path
    image_name = os.path.basename(args.input).split(".")[0]
    os.makedirs(args.output, exist_ok=True)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    for theme in ["French"]:
        edit = f"Transform the image to {theme.replace('_', ' ')} style."
        for cfg_image in args.cfg_image_list:
            for cfg_text in args.cfg_text_list:
                input_image = Image.open(args.input).convert("RGB")
                width, height = input_image.size
                factor = args.resolution / max(width, height)
                factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
                width = int((width * factor) // 64) * 64
                height = int((height * factor) // 64) * 64
                input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

                with torch.no_grad(), autocast("cuda"), model.ema_scope():
                    cond = {}
                    cond["c_crossattn"] = [model.get_learned_conditioning([edit])]
                    input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                    input_image = rearrange(input_image, "h w c -> 1 c h w").to(model.device)
                    cond["c_concat"] = [model.encode_first_stage(input_image).mode()]

                    uncond = {}
                    uncond["c_crossattn"] = [null_token]
                    uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                    sigmas = model_wrap.get_sigmas(args.steps)

                    extra_args = {
                        "cond": cond,
                        "uncond": uncond,
                        "text_cfg_scale": cfg_text,
                        "image_cfg_scale": cfg_image,
                    }
                    torch.manual_seed(seed)
                    z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                    z = sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                    x = model.decode_first_stage(z)
                    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                    x = 255.0 * rearrange(x, "1 c h w -> h w c")
                    edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                edited_image.save(os.path.join(args.output, image_name + f"_{theme}_cfg_text_{cfg_text}_img_{cfg_image}.jpg"))


if __name__ == "__main__":
    main()
