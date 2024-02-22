from __future__ import annotations

from argparse import ArgumentParser

import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from pytorch_lightning import seed_everything
import sys
sys.path.append("..")
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.util import instantiate_from_config
from constants.const import theme_available, class_available

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def main():
    parser = ArgumentParser()
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate_sd.yaml", type=str)
    parser.add_argument("--theme", required=True, type=str)
    parser.add_argument("--cfg-text", default=9.0, type=float)
    parser.add_argument("--seed", type=int, default=[188], nargs="+")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space", )
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    args = parser.parse_args()

    # OPTML1
    args.ckpt = f"../mu_concept_ablation_ca/logs/{args.theme}/checkpoints/last.ckpt"
    args.output_dir = f"eval_results/mu_results/ca/style50/{args.theme}/"

    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}")
    device = "cuda"
    model = model.to(device)
    sampler = DDIMSampler(model)

    print(f"Saving generated images to {args.output_dir}")

    # Extract the file name from a path
    os.makedirs(args.output_dir, exist_ok=True)

    for seed in args.seed:
        seed_everything(seed)
        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
                cfg_text = args.cfg_text
                with torch.no_grad():
                    with autocast("cuda"):
                        with model.ema_scope():
                            uc = model.get_learned_conditioning([""])
                            c = model.get_learned_conditioning(prompt)
                            shape = [4, args.H // 8, args.W // 8]  # downsampling factor 8
                            samples_ddim, _ = sampler.sample(S=args.steps, conditioning=c, batch_size=1, shape=shape,
                                                             verbose=False, unconditional_guidance_scale=cfg_text,
                                                             unconditional_conditioning=uc, eta=args.ddim_eta, x_T=None)

                            x_samples_ddim = model.decode_first_stage(samples_ddim)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1)

                            assert len(x_samples_ddim) == 1

                            x_sample = x_samples_ddim[0]

                            x_sample = (255. * x_sample.numpy()).round()
                            x_sample = x_sample.astype(np.uint8)
                            img = Image.fromarray(x_sample)
                            img.save(os.path.join(args.output_dir, f"{test_theme}_{object_class}_seed{seed}.jpg"))


if __name__ == "__main__":
    main()
