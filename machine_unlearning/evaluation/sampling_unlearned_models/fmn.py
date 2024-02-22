import os.path

import torch
from diffusers import StableDiffusionPipeline
from argparse import ArgumentParser
import sys
sys.path.append("..")
from constants.const import theme_available, class_available


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--theme", required=True, type=str)
    parser.add_argument("--cfg-text-list", default=[9.0], nargs="+", type=float)
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space", )
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--steps", default=100, type=int)

    args = parser.parse_args()

    # OPTML1
    args.ckpt = f"../mu_forget_me_not_fgm/results/style50/{args.theme}"
    args.output_dir = f"eval_results/mu_results/fmn/style50/{args.theme}"
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

    for test_theme in theme_available:
        for object_class in class_available:
            for cfg_text in args.cfg_text_list:
                output_path = os.path.join(args.output_dir, f"{test_theme}_{object_class}_seed{args.seed}.jpg")
                if os.path.exists(output_path):
                    print(f"Detected! Skipping: {output_path}!")
                    continue
                prompt = f"A {object_class} image in {test_theme} style"
                print(f"Generating: {prompt}")
                image = pipe(prompt=prompt, width=args.W, height=args.H, num_inference_steps=args.steps, guidance_scale=cfg_text).images[0]
                image.save(output_path)