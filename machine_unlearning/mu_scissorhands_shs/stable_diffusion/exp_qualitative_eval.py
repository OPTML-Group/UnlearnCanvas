from __future__ import annotations

import json
import math
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
import einops
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

from edm_sampler.external import CompVisDenoiser
from edm_sampler.sampling import sample_euler_ancestral

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])], }
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
        sd = {k: vae_sd[k[len("first_stage_model."):]] if k.startswith("first_stage_model.") else v for k, v in
            sd.items()}
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
    parser.add_argument("--data-path", default="../data/clip-filtered-dataset/", type=str)
    parser.add_argument("--resolution", default=512, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--identifier", required=True, type=str)
    parser.add_argument("--cfg-text-list", default=[3.5, 5.5, 7.5, 9.5, 11.5], type=float, nargs="+")
    parser.add_argument("--cfg-image-list", default=[1.5], type=float, nargs="+")
    parser.add_argument("--seed", type=int, default=10086)
    parser.add_argument("--sample-num", type=int, default=200)
    parser.add_argument("--eval-type", default="edit",
                        # choices=["edit", "depth", "hed", "seg", "depth_inv", "seg", "hed_inv"],
                        choices=["edit", "depth", "hed", "seg"], type=str)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    with open(Path(args.data_path, "seeds.json")) as f:
        seeds = json.load(f)

    total = len(seeds)
    i_start = int(total * 0.9)
    i_end = i_start + args.sample_num

    output_dir = f"imgs/qualitative/{args.identifier}"
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, args.eval_type)

    for i in range(i_start, i_end):
        print(f"===========================> Processing {i}/{total} <===========================")
        name, i_seeds = seeds[i]
        output_sub_dir = os.path.join(output_dir, name)
        os.makedirs(output_sub_dir, exist_ok=True)
        propt_dir = Path(args.data_path, name)
        if args.eval_type == "edit":
            with open(propt_dir.joinpath("prompt.json")) as fp:
                edit_instruction = json.load(fp)["edit"]
        elif args.eval_type == "depth":
            edit_instruction = "Transfer to a depth map"
        elif args.eval_type == "hed":
            edit_instruction = "Transfer to a hed map"
        elif args.eval_type == "seg":
            edit_instruction = "Transfer to a segmentation map"
        else:
            raise NotImplementedError

        image_seed = i_seeds[0]

        input_image_path = propt_dir.joinpath(f"{image_seed}_0.jpg")
        input_image = Image.open(input_image_path).convert("RGB")
        width, height = input_image.size
        factor = args.resolution / max(width, height)
        factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
        width = int((width * factor) // 64) * 64
        height = int((height * factor) // 64) * 64
        input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)

        for cfg_text in args.cfg_text_list:
            for cfg_image in args.cfg_image_list:
                output_image_path = os.path.join(output_sub_dir, f"{image_seed}_text{cfg_text}_image{cfg_image}.jpg")
                with torch.no_grad(), autocast("cuda"), model.ema_scope():
                    cond = {}
                    cond["c_crossattn"] = [model.get_learned_conditioning([edit_instruction])]
                    new_input_image = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
                    new_input_image = rearrange(new_input_image, "h w c -> 1 c h w").to(model.device)
                    cond["c_concat"] = [model.encode_first_stage(new_input_image).mode()]

                    uncond = {}
                    uncond["c_crossattn"] = [null_token]
                    uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

                    sigmas = model_wrap.get_sigmas(args.steps)
                    print(f"Editing the image {input_image_path}, with cfg_text={cfg_text}, cfg_image={cfg_image}")
                    extra_args = {"cond": cond, "uncond": uncond, "text_cfg_scale": cfg_text,
                        "image_cfg_scale": cfg_image, }
                    z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
                    z = sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
                    x = model.decode_first_stage(z)
                    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
                    x = 255.0 * rearrange(x, "1 c h w -> h w c")
                    edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
                edited_image.save(output_image_path)


if __name__ == "__main__":
    main()
