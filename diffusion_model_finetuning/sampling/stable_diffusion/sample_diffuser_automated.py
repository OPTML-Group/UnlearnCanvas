import os.path

import torch
from diffusers import StableDiffusionPipeline
from argparse import ArgumentParser
from pytorch_lightning import seed_everything

import sys
sys.path.append(".")
from constants.const import theme_available, class_available


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--cfg-text", default=9.0, type=float)
    parser.add_argument("--seed", type=int, default=[188, 288, 588, 688, 888], nargs="+")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space", )
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, torch_dtype=torch.float16).to("cuda")

    def dummy(images, **kwargs):
        return images, [False]

    # Disable NSFW checker
    pipe.safety_checker = dummy

    for seed in args.seed:
        seed_everything(seed)
        for theme in theme_available:
            for object_class in class_available:
                cfg_text = args.cfg_text
                if theme == "Seed_Images":
                    prompt = f"A {object_class} image in photo style"
                else:
                    prompt = f"A {object_class} image in {theme.replace('_', ' ')} style"
                print(f"Generating: {prompt}")
                image = pipe(prompt=prompt, width=args.W, height=args.H, num_inference_steps=args.steps, guidance_scale=cfg_text).images[0]
                image.save(os.path.join(args.output_dir, f"{theme}_{object_class}_seed{args.seed}.jpg"))
                print(f"Saved {theme}_{object_class}_seed{args.seed}.jpg")