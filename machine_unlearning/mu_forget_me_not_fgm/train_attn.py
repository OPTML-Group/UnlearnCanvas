import itertools
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.utils.checkpoint
from PIL import Image
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


class ForgetMeNotDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            tokenizer,
            size=512,
            center_crop=False,
            use_added_token=False,
            use_pooler=False,
            multi_concept=None
    ):
        self.use_added_token = use_added_token
        self.use_pooler = use_pooler
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        self.instance_images_path = []
        self.instance_prompt = []

        print(f"***************************************", multi_concept, "***************************************")

        token_idx = 1
        for c, t, num_tok in multi_concept:
            p = Path("data", c)
            if not p.exists():
                raise ValueError(f"Instance {p} images root doesn't exists.")

            image_paths = list(p.iterdir())
            # print(f"***************************************", image_paths, "***************************************")
            self.instance_images_path += image_paths

            target_snippet = f"{''.join([f'<s{token_idx + i}>' for i in range(num_tok)])}" if use_added_token else c.replace(
                "-", " ")
            if t == "object":
                self.instance_prompt += [(f"a {target_snippet} image", target_snippet)] * len(image_paths)
            elif t == "style":
                self.instance_prompt += [(f"an image in {target_snippet} Style", target_snippet)] * len(
                    image_paths)
            else:
                raise ValueError("unknown concept type!")
            if use_added_token:
                token_idx += num_tok
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        instance_prompt, target_tokens = self.instance_prompt[index % self.num_instance_images]

        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_prompt"] = instance_prompt
        example["instance_images"] = self.image_transforms(instance_image)

        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        prompt_ids = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length
        ).input_ids

        concept_ids = self.tokenizer(
            target_tokens,
            add_special_tokens=False
        ).input_ids

        pooler_token_id = self.tokenizer(
            "<|endoftext|>",
            add_special_tokens=False
        ).input_ids[0]

        concept_positions = [0] * self.tokenizer.model_max_length
        for i, tok_id in enumerate(prompt_ids):
            if tok_id == concept_ids[0] and prompt_ids[i:i + len(concept_ids)] == concept_ids:
                concept_positions[i:i + len(concept_ids)] = [1] * len(concept_ids)
            if self.use_pooler and tok_id == pooler_token_id:
                concept_positions[i] = 1
        example["concept_positions"] = torch.tensor(concept_positions)[None]

        return example


def collate_fn(examples, with_prior_preservation=False):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    concept_positions = [example["concept_positions"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    instance_prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = torch.cat(input_ids, dim=0)
    concept_positions = torch.cat(concept_positions, dim=0).type(torch.BoolTensor)

    batch = {
        "instance_prompts": instance_prompts,
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "concept_positions": concept_positions
    }
    return batch


def main(multi_concepts=[["elon-musk", "object"]],
         output_dir="exps_attn/elon-musk",
         pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
         use_ti=True,
         ti_weight_path='results/Bricks/step_inv_500.safetensors',
         only_optimize_ca=False,
         use_pooler=True,
         train_batch_size=1,
         learning_rate=2.0e-06,
         max_train_steps=35,
         revision=None,
         tokenizer_name=None,
         no_real_image=False,
         with_prior_preservation=False,
         seed=None,
         resolution=512,
         center_crop=False,
         train_text_encoder=False,
         num_train_epochs=1,
         checkpointing_steps=500,
         resume_from_checkpoint=None,
         gradient_accumulation_steps=1,
         gradient_checkpointing=False,
         scale_lr=False,
         lr_scheduler="constant",
         lr_warmup_steps=0,
         lr_num_cycles=1,
         lr_power=1.0,
         use_8bit_adam=False,
         dataloader_num_workers=0,
         adam_beta1=0.9,
         adam_beta2=0.999,
         adam_weight_decay=0.01,
         adam_epsilon=1.0e-08,
         max_grad_norm=1.,
         allow_tf32=False,
         mixed_precision="fp16",
         enable_xformers_memory_efficient_attention=False,
         set_grads_to_none=False,
         ):

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )

    if train_text_encoder and gradient_accumulation_steps > 1 and accelerator.num_processes > 1:
        raise ValueError(
            "Gradient accumulation is not supported when training the text encoder in distributed training. "
            "Please set gradient_accumulation_steps to 1. This feature will be supported in the future."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # Load the tokenizer
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=revision, use_fast=False)
    elif pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", revision=revision
    )

    ###

    if use_ti:
        from patch_lora import safe_open, parse_safeloras_embeds, apply_learned_embed_in_clip

        tok_idx = 1
        multi_concept = []
        for c, t in multi_concepts:
            token = None
            idempotent_token = True
            safeloras = safe_open(ti_weight_path, framework="pt", device="cpu")
            tok_dict = parse_safeloras_embeds(safeloras)

            tok_dict = {f"<s{tok_idx + i}>": tok_dict[k] for i, k in enumerate(sorted(tok_dict.keys()))}
            tok_idx += len(tok_dict.keys())
            multi_concept.append([c, t, len(tok_dict.keys())])

            # print("---Adding Tokens---:", c, t)
            apply_learned_embed_in_clip(
                tok_dict,
                text_encoder,
                tokenizer,
                token=token,
                idempotent=idempotent_token,
            )
        multi_concept = multi_concept
    else:
        # dummy number of tok when not using ti
        multi_concept = [[c, t, -1] for c, t in multi_concepts]

    class AttnController:
        def __init__(self) -> None:
            self.attn_probs = []
            self.logs = []

        def __call__(self, attn_prob, m_name) -> Any:
            bs, _ = self.concept_positions.shape
            head_num = attn_prob.shape[0] // bs
            target_attns = attn_prob.masked_select(self.concept_positions[:, None, :].repeat(head_num, 1, 1)).reshape(
                -1, self.concept_positions[0].sum())
            self.attn_probs.append(target_attns)
            self.logs.append(m_name)

        def set_concept_positions(self, concept_positions):
            self.concept_positions = concept_positions

        def loss(self):
            return torch.cat(self.attn_probs).norm()

        def zero_attn_probs(self):
            self.attn_probs = []
            self.logs = []
            self.concept_positions = None

    class MyCrossAttnProcessor:
        def __init__(self, attn_controller: "AttnController", module_name) -> None:
            self.attn_controller = attn_controller
            self.module_name = module_name

        def __call__(self, attn: "CrossAttention", hidden_states, encoder_hidden_states=None, attention_mask=None):
            batch_size, sequence_length, _ = hidden_states.shape
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size=batch_size)

            query = attn.to_q(hidden_states)
            query = attn.head_to_batch_dim(query)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
            key = attn.head_to_batch_dim(key)
            value = attn.head_to_batch_dim(value)

            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            self.attn_controller(attention_probs, self.module_name)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            return hidden_states

    attn_controller = AttnController()
    module_count = 0
    for n, m in unet.named_modules():
        if n.endswith('attn2'):
            m.set_processor(MyCrossAttnProcessor(attn_controller, n))
            module_count += 1
    # print(f"cross attention module count: {module_count}")
    ###

    vae.requires_grad_(False)
    if not train_text_encoder:
        text_encoder.requires_grad_(False)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if train_text_encoder:
            text_encoder.gradient_checkpointing_enable()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        "Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training. copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(unet).dtype != torch.float32:
        raise ValueError(
            f"Unet loaded as datatype {accelerator.unwrap_model(unet).dtype}. {low_precision_error_string}"
        )

    if train_text_encoder and accelerator.unwrap_model(text_encoder).dtype != torch.float32:
        raise ValueError(
            f"Text encoder loaded as datatype {accelerator.unwrap_model(text_encoder).dtype}."
            f" {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if scale_lr:
        learning_rate = (
                learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    if only_optimize_ca:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder
            else [p for n, p in unet.named_parameters()
                  if 'attn2' in n]
        )
        # print("only optimize cross attention...")
    else:
        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if train_text_encoder else unet.parameters()
        )
        # print("optimize unet...")
    optimizer = optimizer_class(
        params_to_optimize,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Dataset and DataLoaders creation:
    train_dataset = ForgetMeNotDataset(
        tokenizer=tokenizer,
        size=resolution,
        center_crop=center_crop,
        use_pooler=use_pooler,
        use_added_token=use_ti,
        multi_concept=multi_concept
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, with_prior_preservation),
        num_workers=dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )

    # Prepare everything with our `accelerator`.
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    if not train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    if overrode_max_train_steps:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("forgetmenot")

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    debug_once = True
    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        if train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # show
                if debug_once:
                    print(batch["instance_prompts"][0])
                    debug_once = False
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if no_real_image:
                    noisy_latents = noise_scheduler.add_noise(torch.zeros_like(noise), noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # set concept_positions for this batch 
                attn_controller.set_concept_positions(batch["concept_positions"])

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                ### collect attentions prob
                loss = attn_controller.loss()
                ###

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = params_to_optimize
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=set_grads_to_none)
                attn_controller.zero_attn_probs()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        pipeline = DiffusionPipeline.from_pretrained(
            pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=accelerator.unwrap_model(text_encoder),
            tokenizer=tokenizer,
            revision=revision,
        )
        pipeline.save_pretrained(output_dir)
        # if isinstance(args, Namespace):
        #     with open(f"{output_dir}/my_args.json", "w") as f:
        #         json.dump(vars(args), f, indent=4)

    accelerator.end_training()


import argparse
from constants.const import class_available, theme_available



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_path', type=str,
                        default='../main_sd_image_editing/ckpts/sd_model/diffuser/style38/step4999/')
    parser.add_argument('--theme', type=str, choices=class_available + theme_available)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--ti_weight_path', type=str, default=None)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max-steps', type=int, default=100)
    parser.add_argument('--only_xa', action='store_true')
    args = parser.parse_args()

    args.multi_concept = [[args.theme, "object"] if args.theme in class_available else [args.theme, "style"]]
    args.output_dir = os.path.join(args.output_dir, args.theme)
    if args.ti_weight_path is None:
        args.ti_weight_path = os.path.join(args.output_dir, "step_inv_500.safetensors")
        print("ti_weight_path is None, use default path: ", args.ti_weight_path)
        if not os.path.exists(args.ti_weight_path):
            raise ValueError("ti_weight_path is None, but not exist")

    main(multi_concepts=args.multi_concept,
         output_dir=args.output_dir,
         pretrained_model_name_or_path=args.pretrained_path,
         use_ti=True,
         ti_weight_path=args.ti_weight_path,
         learning_rate=args.lr,
         max_train_steps=args.max_steps,
         only_optimize_ca=args.only_xa,
         )
