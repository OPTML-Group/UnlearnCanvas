import os

from diffusers.pipelines import BlipDiffusionControlNetPipeline
from diffusers.utils import load_image
from controlnet_aux import CannyDetector
import torch
import argparse
from tqdm import tqdm
from constants.const import theme_available, class_available

from time import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Artistic Test')

    parser.add_argument('--test_theme', default=None, type=str, nargs='+')
    parser.add_argument('--img_dir', required=True, type=str)
    parser.add_argument('--output_dir', required=True, type=str)

    args = parser.parse_args()

    blip_diffusion_pipe = BlipDiffusionControlNetPipeline.from_pretrained(
        "Salesforce/blipdiffusion-controlnet", torch_dtype=torch.float16, cache_dir="./cache"
    ).to("cuda")

    guidance_scale = 7.5
    num_inference_steps = 100
    negative_prompt = "over-exposure, under-exposure, saturated, duplicate, out of frame, lowres, cropped, worst quality, low quality, jpeg artifacts, morbid, mutilated, out of frame, ugly, bad anatomy, bad proportions, deformed, blurry, duplicate"

    test_theme = theme_available if args.test_theme is None else args.test_theme
    for theme in tqdm(test_theme):
        assert theme in theme_available

    for theme in tqdm(test_theme):
        if theme == "Seed_Images":
            continue
        output_dir = os.path.join(args.output_dir, theme)
        os.makedirs(output_dir, exist_ok=True)
        for object_class in class_available:
            style_subject = object_class
            tgt_subject = object_class
            text_prompt = f"a {object_class} image"
            for test_img_idx in [19, 20]:
                for style_img_idx in range(1, 19):
                    start_time = time()
                    content_img = os.path.join(args.img_dir, "Seed_Images", object_class, str(test_img_idx) + '.jpg')
                    style_img = os.path.join(args.img_dir, theme, object_class, str(style_img_idx) + '.jpg')
                    output_path = os.path.join(output_dir, f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")

                    cldm_cond_image = load_image(
                        content_img
                    ).resize((512, 512))
                    canny = CannyDetector()
                    cldm_cond_image = canny(cldm_cond_image, 30, 70, output_type="pil")
                    style_image = load_image(
                        style_img
                    )

                    output = blip_diffusion_pipe(
                        text_prompt,
                        style_image,
                        cldm_cond_image,
                        style_subject,
                        tgt_subject,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        neg_prompt=negative_prompt,
                        height=512,
                        width=512,
                    ).images
                    output[0].save(output_path)
                    print(f"Saved to {output_path}")

                    print(f"Time taken: {time() - start_time: .2f}s")
