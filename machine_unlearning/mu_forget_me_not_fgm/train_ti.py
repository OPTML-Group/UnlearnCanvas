# Bootstrapped from:
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/cli_lora_pti.py

import itertools
import math
import os
import re
from typing import Optional, List, Literal
import argparse
from constants.const import class_available, theme_available

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import wandb

from lora_diffusion import (
    inspect_lora,
    save_all,
    prepare_clip_model_sets,
    evaluate_pipe,
)

from dataset import PivotalTuningDatasetCapation

def get_models(
    pretrained_model_name_or_path,
    pretrained_vae_name_or_path,
    revision,
    placeholder_tokens: List[str],
    initializer_tokens: List[str],
    device="cuda:0",
):

    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=revision,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )

    placeholder_token_ids = []

    token_list = []
    for init_tok in initializer_tokens:
        token_ids = tokenizer.encode(init_tok)
        token_list = token_list + token_ids
    assert len(token_list) <= len(placeholder_tokens)

    for idx, token in enumerate(placeholder_tokens):
        num_added_tokens = tokenizer.add_tokens(token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        placeholder_token_id = tokenizer.convert_tokens_to_ids(token)

        placeholder_token_ids.append(placeholder_token_id)

        # Load models and create wrapper for stable diffusion

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data

        if idx < len(token_list):
            token_embeds[placeholder_token_id] = token_embeds[token_list[idx]]
        else:
            init_tok = "<rand-1>"
            # <rand-"sigma">, e.g. <rand-0.5>
            sigma_val = float(re.findall(r"<rand-(.*)>", init_tok)[0])

            token_embeds[placeholder_token_id] = (
                torch.randn_like(token_embeds[0]) * sigma_val
            )
            print(
                f"Initialized {token} with random noise (sigma={sigma_val}), empirically {token_embeds[placeholder_token_id].mean().item():.3f} +- {token_embeds[placeholder_token_id].std().item():.3f}"
            )
            print(f"Norm : {token_embeds[placeholder_token_id].norm():.4f}")

    vae = AutoencoderKL.from_pretrained(
        pretrained_vae_name_or_path or pretrained_model_name_or_path,
        subfolder=None if pretrained_vae_name_or_path else "vae",
        revision=None if pretrained_vae_name_or_path else revision,
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet",
        revision=revision,
    )

    return (
        text_encoder.to(device),
        vae.to(device),
        unet.to(device),
        tokenizer,
        placeholder_token_ids,
    )


def text2img_dataloader(train_dataset, train_batch_size, tokenizer, vae, text_encoder):
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        uncond_ids = [example["uncond_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if examples[0].get("class_prompt_ids", None) is not None:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        uncond_ids = tokenizer.pad(
            {"input_ids": uncond_ids},
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "uncond_ids":uncond_ids,
            "pixel_values": pixel_values,
        }

        if examples[0].get("mask", None) is not None:
            batch["mask"] = torch.stack([example["mask"] for example in examples])

        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    return train_dataloader


def loss_step(
    batch,
    unet,
    vae,
    text_encoder,
    scheduler,
    t_mutliplier=1.0,
    mixed_precision=True,
):
    weight_dtype = torch.float32

    latents = vae.encode(
        batch["pixel_values"].to(dtype=weight_dtype).to(unet.device)
    ).latent_dist.sample()
    latents = latents * 0.18215

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]

    timesteps = torch.randint(
        0,
        int(scheduler.config.num_train_timesteps * t_mutliplier),
        (bsz,),
        device=latents.device,
    )
    timesteps = timesteps.long()

    noisy_latents = scheduler.add_noise(latents, noise, timesteps)

    if mixed_precision:
        with torch.cuda.amp.autocast():

            encoder_hidden_states = text_encoder(
                batch["input_ids"].to(text_encoder.device)
            )[0]

            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    else:

        encoder_hidden_states = text_encoder(
            batch["input_ids"].to(text_encoder.device)
        )[0]

        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        # noise_pred_uncond, noise_pred_text = model_pred.chunk(2)
        # model_pred = noise_pred_uncond + 7 * (noise_pred_text - noise_pred_uncond)

    if scheduler.config.prediction_type == "epsilon":
        target = noise
    elif scheduler.config.prediction_type == "v_prediction":
        target = scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {scheduler.config.prediction_type}")

    if batch.get("mask", None) is not None:
        mask = (
            batch["mask"]
            .to(model_pred.device)
            .reshape(
                model_pred.shape[0], 1, batch["mask"].shape[2], batch["mask"].shape[3]
            )
        )
        # resize to match model_pred
        mask = (
            F.interpolate(
                mask.float(),
                size=model_pred.shape[-2:],
                mode="nearest",
            )
            + 0.05
        )

        mask = mask / mask.mean()

        model_pred = model_pred * mask

        target = target * mask

    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    return loss


def train_inversion(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps: int,
    scheduler,
    index_no_updates,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path: str,
    tokenizer,
    lr_scheduler,
    test_image_path: str,
    accum_iter: int = 1,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    class_token: str = "person",
    mixed_precision: bool = True,
    clip_ti_decay: bool = True,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    # Original Emb for TI
    orig_embeds_params = text_encoder.get_input_embeddings().weight.data.clone()

    if log_wandb:
        preped_clip = prepare_clip_model_sets()

    index_updates = ~index_no_updates
    loss_sum = 0.0

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        unet.eval()
        text_encoder.train()
        for batch in dataloader:

            lr_scheduler.step()

            with torch.set_grad_enabled(True):
                loss = (
                    loss_step(
                        batch,
                        unet,
                        vae,
                        text_encoder,
                        scheduler,
                        mixed_precision=mixed_precision,
                    )
                    / accum_iter
                )

                loss.backward()
                loss_sum += loss.detach().item()

                if global_step % accum_iter == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():

                        # normalize embeddings
                        if clip_ti_decay:
                            pre_norm = (
                                text_encoder.get_input_embeddings()
                                .weight[index_updates, :]
                                .norm(dim=-1, keepdim=True)
                            )

                            lambda_ = min(1.0, 100 * lr_scheduler.get_last_lr()[0])
                            text_encoder.get_input_embeddings().weight[
                                index_updates
                            ] = F.normalize(
                                text_encoder.get_input_embeddings().weight[
                                    index_updates, :
                                ],
                                dim=-1,
                            ) * (
                                pre_norm + lambda_ * (0.4 - pre_norm)
                            )
                            # print(pre_norm)

                        current_norm = (
                            text_encoder.get_input_embeddings()
                            .weight[index_updates, :]
                            .norm(dim=-1)
                        )

                        text_encoder.get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                        # print(f"Current Norm : {current_norm}")

                global_step += 1
                progress_bar.update(1)

                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

            if global_step % save_steps == 0:
                save_all(
                    unet=unet,
                    text_encoder=text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_inv_{global_step}.safetensors"
                    ),
                    save_lora=False,
                )
                if log_wandb:
                    with torch.no_grad():
                        pipe = StableDiffusionPipeline(
                            vae=vae,
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            unet=unet,
                            scheduler=scheduler,
                            safety_checker=None,
                            feature_extractor=None,
                        )

                        # open all images in test_image_path
                        images = []
                        for file in os.listdir(test_image_path):
                            if file.endswith(".png") or file.endswith(".jpg"):
                                images.append(
                                    Image.open(os.path.join(test_image_path, file))
                                )

                        wandb.log({"loss": loss_sum / save_steps})
                        loss_sum = 0.0
                        wandb.log(
                            evaluate_pipe(
                                pipe,
                                target_images=images,
                                class_token=class_token,
                                learnt_token="".join(placeholder_tokens),
                                n_test=wandb_log_prompt_cnt,
                                n_step=50,
                                clip_model_sets=preped_clip,
                            )
                        )

            if global_step >= num_steps:
                return


def perform_tuning(
    unet,
    vae,
    text_encoder,
    dataloader,
    num_steps,
    scheduler,
    optimizer,
    save_steps: int,
    placeholder_token_ids,
    placeholder_tokens,
    save_path,
    lr_scheduler_lora,
):

    progress_bar = tqdm(range(num_steps))
    progress_bar.set_description("Steps")
    global_step = 0

    weight_dtype = torch.float16

    unet.train()
    text_encoder.train()

    for epoch in range(math.ceil(num_steps / len(dataloader))):
        for batch in dataloader:
            lr_scheduler_lora.step()

            optimizer.zero_grad()

            loss = loss_step(
                batch,
                unet,
                vae,
                text_encoder,
                scheduler,
                t_mutliplier=0.8,
                mixed_precision=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                itertools.chain(unet.parameters(), text_encoder.parameters()), 1.0
            )
            optimizer.step()
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler_lora.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            global_step += 1

            if global_step % save_steps == 0:
                save_all(
                    unet,
                    text_encoder,
                    placeholder_token_ids=placeholder_token_ids,
                    placeholder_tokens=placeholder_tokens,
                    save_path=os.path.join(
                        save_path, f"step_{global_step}.safetensors"
                    ),
                )
                moved = (
                    torch.tensor(list(itertools.chain(*inspect_lora(unet).values())))
                    .mean()
                    .item()
                )

                # print("LORA Unet Moved", moved)
                moved = (
                    torch.tensor(
                        list(itertools.chain(*inspect_lora(text_encoder).values()))
                    )
                    .mean()
                    .item()
                )

                # print("LORA CLIP Moved", moved)

            if global_step >= num_steps:
                return


def train(
    instance_data_dir: str,
    pretrained_model_name_or_path: str = "../main_sd_image_editing/ckpts/sd_model/diffuser/style38/step4999/",
    output_dir: str = "results/",
    pretrained_vae_name_or_path: str = None,
    revision: Optional[str] = None,
    class_data_dir: Optional[str] = None,
    perform_inversion: bool = True,
    use_template: Literal[None, "object", "style", "naked"] = None,
    placeholder_tokens: str = "<s1>|<s2>|<s3>|<s4>|<s5>|<s6>|<s7>|<s8>|<s9>|<s10>",
    placeholder_token_at_data: Optional[str] = "<s>|<s1><s2><s3><s4><s5><s6><s7><s8><s9><s10>",
    initializer_tokens: Optional[str] = "Bricks",
    class_prompt: Optional[str] = None,
    with_prior_preservation: bool = False,
    seed: int = 42,
    resolution: int = 512,
    color_jitter: bool = False,
    train_batch_size: int = 1,
    max_train_steps_ti: int = 500,
    save_steps: int = 500,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    clip_ti_decay: bool = True,
    learning_rate_ti: float = 0.001,
    use_face_segmentation_condition: bool = False,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 100,
    weight_decay_ti: float = 0.1,
    device="cuda:0",
    extra_args: Optional[dict] = None,
    log_wandb: bool = False,
    wandb_log_prompt_cnt: int = 10,
    wandb_project_name: str = "new_pti_project",
    wandb_entity: str = "new_pti_entity",
):
    torch.manual_seed(seed)

    if log_wandb:
        wandb.init(
            project=wandb_project_name,
            entity=wandb_entity,
            name=f"steps_{max_train_steps_ti}_lr_{learning_rate_ti}_{instance_data_dir.split('/')[-1]}",
            reinit=True,
            config={
                "lr": learning_rate_ti,
                **(extra_args if extra_args is not None else {}),
            },
        )

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    # print(placeholder_tokens, initializer_tokens)
    placeholder_tokens = placeholder_tokens.split("|")
    if initializer_tokens is None:
        print("PTI : Initializer Token not give, random inits")
        initializer_tokens = ["<rand-0.017>"] * len(placeholder_tokens)
    else:
        initializer_tokens = initializer_tokens.split("|")

    # assert len(initializer_tokens) == len(
    #     placeholder_tokens
    # ), "Unequal Initializer token for Placeholder tokens."

    class_token = "".join(initializer_tokens)

    if placeholder_token_at_data is not None:
        tok, pat = placeholder_token_at_data.split("|")
        token_map = {tok: pat}

    else:
        token_map = {"DUMMY": "".join(placeholder_tokens)}

    print("Placeholder Tokens", placeholder_tokens)
    print("Initializer Tokens", initializer_tokens)

    # get the models
    text_encoder, vae, unet, tokenizer, placeholder_token_ids = get_models(
        pretrained_model_name_or_path,
        pretrained_vae_name_or_path,
        revision,
        placeholder_tokens,
        initializer_tokens,
        device=device,
    )

    noise_scheduler = DDPMScheduler.from_config(
        pretrained_model_name_or_path, subfolder="scheduler"
    )

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if scale_lr:
        ti_lr = learning_rate_ti * gradient_accumulation_steps * train_batch_size
    else:
        ti_lr = learning_rate_ti

    train_dataset = PivotalTuningDatasetCapation(
        instance_data_root=instance_data_dir,
        token_map=token_map,
        use_template=use_template,
        class_data_root=class_data_dir if with_prior_preservation else None,
        class_prompt=class_prompt,
        tokenizer=tokenizer,
        size=resolution,
        color_jitter=color_jitter,
        use_face_segmentation_condition=use_face_segmentation_condition,
    )

    train_dataset.blur_amount = 20

    train_dataloader = text2img_dataloader(
        train_dataset, train_batch_size, tokenizer, vae, text_encoder
    )

    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_ids[0]

    for tok_id in placeholder_token_ids:
        index_no_updates[tok_id] = False

    unet.requires_grad_(False)
    vae.requires_grad_(False)

    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    for param in params_to_freeze:
        param.requires_grad = False

    # STEP 1 : Perform Inversion
    if perform_inversion:
        ti_optimizer = optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),
            lr=ti_lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay_ti,
        )

        lr_scheduler = get_scheduler(
            lr_scheduler,
            optimizer=ti_optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=max_train_steps_ti,
        )

        train_inversion(
            unet,
            vae,
            text_encoder,
            train_dataloader,
            max_train_steps_ti,
            accum_iter=gradient_accumulation_steps,
            scheduler=noise_scheduler,
            index_no_updates=index_no_updates,
            optimizer=ti_optimizer,
            lr_scheduler=lr_scheduler,
            save_steps=save_steps,
            placeholder_tokens=placeholder_tokens,
            placeholder_token_ids=placeholder_token_ids,
            save_path=output_dir,
            test_image_path=instance_data_dir,
            log_wandb=log_wandb,
            wandb_log_prompt_cnt=wandb_log_prompt_cnt,
            class_token=class_token,
            mixed_precision=False,
            tokenizer=tokenizer,
            clip_ti_decay=clip_ti_decay,
        )

        del ti_optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Forget Me Not - Train TI")
    parser.add_argument('--instance_data_dir', type=str, default='data/')
    parser.add_argument('--pretrained_path', type=str, default='../main_sd_image_editing/ckpts/sd_model/diffuser/style50/step19999/')
    parser.add_argument('--theme', type=str, choices= class_available + theme_available)
    parser.add_argument('--output_dir', type=str, default='results/style50/')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    args.instance_data_dir = os.path.join(args.instance_data_dir, args.theme)
    template = 'object' if args.theme in class_available else 'style'
    args.output_dir = os.path.join(args.output_dir, args.theme)
    initializer_tokens = f"{args.theme}"
    train(instance_data_dir=args.instance_data_dir,
          pretrained_model_name_or_path=args.pretrained_path,
          use_template=template, output_dir=args.output_dir,
          initializer_tokens=initializer_tokens,
          max_train_steps_ti=args.steps, save_steps=args.steps, learning_rate_ti=args.lr)