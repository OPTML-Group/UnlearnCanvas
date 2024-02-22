from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
import torch
from PIL import Image
import argparse
import os
import sys
sys.path.append("..")

from constants.const import theme_available, class_available

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--theme', type=str, required=True, choices=theme_available + class_available)
    # parser.add_argument('--ckpt', help='path to the unet ckpt', type=str, required=True)
    parser.add_argument('--pipeline_path', help='path to pipeline', type=str, default="ckpts/sd_model/diffuser/style50/step19999/")
    parser.add_argument('--output_dir', help='folder where to save images', type=str, default="eval_results/mu_results/uce/style50/")
    parser.add_argument('--seed', help='seed', type=int, required=False, default=188)
    parser.add_argument('--cfg_txt', help='guidance to run eval', type=float, required=False, default=9.0)
    parser.add_argument('--steps', help='ddim steps of inference used to train', type=int, required=False,
                        default=100)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(args.output_dir, exist_ok=True)

    args.ckpt = os.path.join(f"../mu_unified_concept_editing_uce/results/style50/", args.theme)

    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = AutoencoderKL.from_pretrained(args.pipeline_path, subfolder="vae", cache_dir="./cache", torch_dtype=torch.float16)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(args.pipeline_path, subfolder="tokenizer", cache_dir="./cache", torch_dtype=torch.float16)
    text_encoder = CLIPTextModel.from_pretrained(args.pipeline_path, subfolder="text_encoder", cache_dir="./cache", torch_dtype=torch.float16)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(args.pipeline_path, subfolder="unet", cache_dir="./cache", torch_dtype=torch.float16)
    unet.load_state_dict(torch.load(args.ckpt, map_location=device))
    unet.to(torch.float16)
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

    vae.to(device)
    text_encoder.to(device)
    unet.to(device)

    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    batch_size = 1

    for test_theme in theme_available:
        for object_class in class_available:
            output_path = f"{args.output_dir}/{test_theme}_{object_class}_seed{args.seed}.jpg"
            if os.path.exists(output_path):
                print(f"Detected! Skipping {output_path}")
                continue
            prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
            generator = torch.manual_seed(args.seed)  # Seed generator to create the inital latent noise
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                                   return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

            max_length = text_input.input_ids.shape[-1]
            uncond_input = tokenizer(
                [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
            )
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            latents = torch.randn(
                (batch_size, unet.in_channels, height // 8, width // 8),
                generator=generator,
            )
            latents = latents.to(device)

            scheduler.set_timesteps(args.steps)

            latents = latents * scheduler.init_noise_sigma

            from tqdm.auto import tqdm
            scheduler.set_timesteps(args.steps)
            # the model is trained in fp16, use mixed precision forward pass
            with torch.cuda.amp.autocast():
                # predict the noise residual
                with torch.no_grad():
                    for t in tqdm(scheduler.timesteps):
                        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                        latent_model_input = torch.cat([latents] * 2)

                        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                        # perform guidance
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.cfg_txt * (noise_pred_text - noise_pred_uncond)

                        # compute the previous noisy sample x_t -> x_t-1
                        latents = scheduler.step(noise_pred, t, latents).prev_sample

            # the model is trained in fp16, use mixed precision forward pass
            with torch.cuda.amp.autocast():
                # scale and decode the image latents with vae
                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = vae.decode(latents).sample

                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images][0]
            pil_images.save(output_path)
