from typing import Optional, Union, Tuple, List, Dict  
from tqdm import tqdm  
import torch  
from diffusers import StableDiffusionPipeline, DDIMScheduler  
import numpy as np  
import abc  
import time  
from PIL import Image 
import os  
import argparse  
import ast  
from scipy.spatial.distance import cdist  
import torch.nn.functional as nnf
from torch.optim.adam import Adam  
from torch import nn  
from torch.nn import functional as F
import importlib

import sys
sys.path.append(".")

from constants.const import theme_available, class_available
from pytorch_lightning import seed_everything
import wandb
  
CALC_SIMILARITY = True  
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False, steps_offset=1)
LOW_RESOURCE = True
MAX_NUM_WORDS = 77

GUIDANCE_SCALE = 9.0
NUM_DDIM_STEPS = 50  
GUIDANCE_SCALE_TRAIN = 9.0

class Loss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(Loss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)

class AttnLoss(nn.Module):
    def __init__(self, device, attn_loss_type, n, token_indices, lambda_retain=1., lambda_erase=-1., lambda_self_retain=1., lambda_self_erase=-1.):
        super(AttnLoss, self).__init__()
        self.device = device
        self.prompt_n = n
        self.token_indices = token_indices

        self.lambda_retain = lambda_retain
        self.lambda_erase = lambda_erase
        self.lambda_self_retain = lambda_self_retain
        self.lambda_self_erase = lambda_self_erase

        self.retain_loss = Loss(attn_loss_type)
        self.erase_loss = Loss(attn_loss_type)
        self.self_retain_loss = Loss(attn_loss_type)
        self.self_erase_loss = Loss(attn_loss_type)

    def calc_mask(self, attn, threshold=.85):
        mask = []
        for i in [num for num in range(1, self.prompt_n-1)]:
            _attn = attn[:,:,i].clone()
            _attn = 255 * _attn / _attn.max()
            _attn = F.interpolate(_attn.unsqueeze(0).unsqueeze(0), size=(256, 256), mode='bilinear')
            if i in self.token_indices:
                _threshold = threshold
            else:
                _threshold = threshold + .1
            _attn[_attn >= _attn.max() * _threshold] = 255
            _attn[_attn < _attn.max() * _threshold] = 0
            _attn = F.interpolate(_attn, size=attn.shape[:2], mode='bilinear')
            mask += [_attn.squeeze(0).squeeze(0)]
        return mask

    def calc_retain_loss(self, attn, attn_erase):
        loss = .0
        for i in [num for num in range(1, self.prompt_n-1)]:
            if i in self.token_indices:
                continue
            loss += self.retain_loss(attn[:,:,i], attn_erase[:,:,i])
        # print(f'\n retain loss: {loss.item()}, ', end=' ')
        return loss

    def calc_erase_loss(self, attn, attn_erase):
        loss = .0
        for i in self.token_indices:
            loss += self.erase_loss(attn[:,:,i], attn_erase[:,:,i])
        # print(f'erase loss: {loss.item()}')
        return loss

    def calc_self_retain_loss(self, self_attn, self_attn_erase, mask):
        loss = .0
        h, w = mask[0].shape
        for i in [num for num in range(1, self.prompt_n-1)]:
            if i in self.token_indices:
                continue
            for j, m in enumerate(mask[i-1].reshape(h*w)):
                if m > 0:
                    loss += self.self_retain_loss(self_attn[:,:,j].view(-1).unsqueeze(0),
                                                  self_attn_erase[:,:,j].view(-1).unsqueeze(0))
        # print(f'self retain loss: {loss.item()}, ', end=' ')
        return loss

    def calc_self_erase_loss(self, self_attn, self_attn_erase, mask):
        loss = .0
        h, w = mask[0].shape
        for i in self.token_indices:
            for j, m in enumerate(mask[i-1].reshape(h*w)):
                if m > 0:
                    loss += self.self_erase_loss(self_attn[:,:,j].view(-1).unsqueeze(0),
                                                 self_attn_erase[:,:,j].view(-1).unsqueeze(0))
        # print(f'self erase loss: {loss.item()}')
        return loss

    def forward(self, attn, attn_erase, self_attn, self_attn_erase):
        attn, attn_erase, self_attn, self_attn_erase \
            = attn.to(torch.double), attn_erase.to(torch.double), self_attn.to(torch.double), self_attn_erase.to(torch.double)
        attn_loss = .0

        if self.lambda_self_retain or self.lambda_self_erase:
            mask = self.calc_mask(attn)

        h, w, seq_len = attn.shape
        attn = attn.reshape(h*w, seq_len).unsqueeze(0)
        attn_erase = attn_erase.reshape(h*w, seq_len).unsqueeze(0)

        if self.lambda_retain:
            attn_loss += self.lambda_retain * self.calc_retain_loss(attn, attn_erase)

        if self.lambda_erase:
            attn_loss += self.lambda_erase * self.calc_erase_loss(attn, attn_erase)

        if self.lambda_self_retain:
            attn_loss += self.lambda_self_retain * self.calc_self_retain_loss(self_attn, self_attn_erase, mask)

        if self.lambda_self_erase:
            attn_loss += self.lambda_self_erase * self.calc_self_erase_loss(self_attn, self_attn_erase, mask)

        loss = attn_loss
        return loss


class NullInversion:

    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE_TRAIN
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(self.model.device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)
        return image_rec, ddim_latents

    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))
            latent_prev = latents[len(latents) - i - 2]
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE_TRAIN * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
        bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0, 0, 0, 0), num_inner_steps=10, early_stop_epsilon=1e-5,
               inversion='Null-text', verbose=False):
        self.init_prompt(prompt)
        register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)
        if verbose:
            print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        assert inversion in ['NT', 'NPI']
        if inversion == 'NT':
            print("Null-text optimization...")
            uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)
        else:
            print("Negative prompt inversion...")
            uncond_embeddings, cond_embeddings = self.context.chunk(2)
            uncond_embeddings = [cond_embeddings] * NUM_DDIM_STEPS
        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings

    def __init__(self, model):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False,
                                  set_alpha_to_one=False)
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.asarray(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def punish_wight(wo_batch, latent_size, alpha, method):
    if method == 'weight':
        wo_batch *= alpha
    elif method in ['alpha', 'beta', 'delete', 'soft-weight']:
        u, s, vh = torch.linalg.svd(wo_batch)
        u = u[:,:latent_size]
        zero_idx = int(latent_size * alpha)
        if method == 'alpha':
            s[:zero_idx] = 0
        elif method == 'beta':
            s[zero_idx:] = 0
        elif method == 'delete':
            s = s[zero_idx:] if zero_idx < latent_size else torch.zeros(latent_size).to(s.device)
            u = u[:, zero_idx:] if zero_idx < latent_size else u
            vh = vh[zero_idx:, :] if zero_idx < latent_size else vh
        elif method == 'soft-weight':
            if CALC_SIMILARITY:
                _s = s.clone()
                _s[zero_idx:] = 0
                _wo_batch = u @ torch.diag(_s) @ vh
                dist = cdist(wo_batch[:,0].unsqueeze(0).cpu(), _wo_batch[:,0].unsqueeze(0).cpu(), metric='cosine')
                # print(f'The distance between the word embedding before and after the punishment: {dist}')
            if alpha == -.001:
                s *= (torch.exp(-.001 * s) * 1.2)  # strengthen objects (our Appendix.F)
            else:
                s *= torch.exp(-alpha*s)  # suppression EOT (our main paper)

        wo_batch = u @ torch.diag(s) @ vh
    else:
        raise ValueError('Unsupported method')
    return wo_batch

def woword_eot_context(context, token_indices, alpha, method, n):
    for i, batch in enumerate(context):
        indices = token_indices + [num for num in range(n-1, 77)]
        wo_batch = batch[indices]
        wo_batch = punish_wight(wo_batch.T, len(indices), alpha, method).T
        batch[indices] = wo_batch
    return context

def woword_reweight(attn, token_indices, alpha):

    latent_size = int(attn.shape[1]**0.5)
    assert latent_size**2 == attn.shape[1]
    for head_attn in attn:
        for indice in token_indices:
            wo_attn = head_attn[:, indice].reshape(latent_size, latent_size)
            wo_attn *= alpha  # same as Reweight of P2P
            head_attn[:, indice] = wo_attn.reshape(latent_size**2)
    return attn


def view_images(images, num_rows=1, offset_ratio=0.02, save_name="null-text+ptp"):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    # pil_img = Image.fromarray(image_).save(f'{save_name}.png')
    pil_img = Image.fromarray(image_).save(f'{save_name}.jpg')


def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        controller.uncond = True
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        controller.uncond = False
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


def register_attention_control(model, controller):
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None

            residual = hidden_states

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
            attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            query = self.head_to_batch_dim(query)
            key = self.head_to_batch_dim(key)
            value = self.head_to_batch_dim(value)

            attention_probs = self.get_attention_scores(query, key, attention_mask)
            attention_probs = controller(attention_probs, is_cross, place_in_unet)

            hidden_states = torch.bmm(attention_probs, value)
            hidden_states = self.batch_to_head_dim(hidden_states)

            # linear proj
            hidden_states = to_out(hidden_states)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0
            self.ddim_inv = True

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'Attention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net[1], 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net[1], 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net[1], 0, "mid")

    controller.num_att_layers = cross_att_count

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor] = None):
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
            inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
            for i, ind in enumerate(inds):
                if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words


class EmptyControl:

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn

class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class SpatialReplace(EmptyControl):

    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, token_indices: List[int], alpha: float, method: str, cross_retain_steps: float, n: int, iter_each_step: int, max_step_to_erase: int,
                 lambda_retain=1, lambda_erase=-.5, lambda_self_retain=1, lambda_self_erase=-.5):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.baseline = True
        # for suppression content
        self.ddim_inv = False
        self.token_indices = token_indices
        self.uncond = True
        self.alpha = alpha
        self.method = method  # default: 'soft-weight'
        self.i = None
        self.cross_retain_steps = cross_retain_steps * NUM_DDIM_STEPS
        self.n = n
        self.text_embeddings_erase = None
        self.iter_each_step = iter_each_step
        self.MAX_STEP_TO_ERASE = max_step_to_erase
        # lambds of loss
        self.lambda_retain = lambda_retain
        self.lambda_erase = lambda_erase
        self.lambda_self_retain = lambda_self_retain
        self.lambda_self_erase = lambda_self_erase


def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int=0):
    out = []
    attention_maps = attention_store.get_average_attention()

    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(1, -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


# Infernce Code
@torch.no_grad()
def text2image_ldm_stable(
        model,
        prompt: List[str],
        controller,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        generator: Optional[torch.Generator] = None,
        latent: Optional[torch.FloatTensor] = None,
        uncond_embeddings=None,
        start_time=50,
        return_type='image'
):
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
        scale = 20
    else:
        uncond_embeddings_ = None
        scale = 5

    latent, _ = init_latent(latent, model, height, width, generator, batch_size)

    latents = latent.clone().to(model.device)
    attn_loss_func = AttnLoss(model.device, 'cosine', controller.n, controller.token_indices,
                              controller.lambda_retain, controller.lambda_erase, controller.lambda_self_retain, controller.lambda_self_erase)

    model.scheduler.set_timesteps(num_inference_steps)
    # text embedding for erasing
    controller.text_embeddings_erase = text_embeddings.clone()

    scale_range = np.linspace(1., .1, len(model.scheduler.timesteps))
    pbar = tqdm(model.scheduler.timesteps[-start_time:], desc='Suppress EOT', ncols=100, colour="red")
    for i, t in enumerate(pbar):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            if LOW_RESOURCE:
                context = (uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings)
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
            if LOW_RESOURCE:
                context = (uncond_embeddings_, text_embeddings)
        controller.i = i

        # conditional branch: erase content for text embeddings
        if controller.i >= controller.cross_retain_steps:
            controller.text_embeddings_erase = \
                woword_eot_context(text_embeddings.clone(), controller.token_indices, controller.alpha,
                                            controller.method, controller.n)

        controller.baseline = True
        if controller.MAX_STEP_TO_ERASE > controller.i >= controller.cross_retain_steps and not (controller.text_embeddings_erase == text_embeddings).all() and \
                (attn_loss_func.lambda_retain or attn_loss_func.lambda_erase or attn_loss_func.lambda_self_retain or attn_loss_func.lambda_self_erase):
            controller.uncond = False
            controller.cur_att_layer = 32  # w=1, skip unconditional branch
            controller.attention_store = {}
            noise_prediction_text = model.unet(latents, t, encoder_hidden_states=text_embeddings)["sample"]
            attention_maps = aggregate_attention(controller, 16, ["up", "down"], is_cross=True)
            self_attention_maps = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=False)

            del noise_prediction_text
            # update controller.text_embeddings_erase for some timestep
            iter = controller.iter_each_step
            while iter > 0:
                with torch.enable_grad():
                    controller.cur_att_layer = 32  # w=1, skip unconditional branch
                    controller.attention_store = {}
                    # conditional branch
                    text_embeddings_erase = controller.text_embeddings_erase.clone().detach().requires_grad_(True)
                    # forward pass of conditional branch with text_embeddings_erase
                    noise_prediction_text = model.unet(_latent_erase, t, encoder_hidden_states=text_embeddings_erase)["sample"]
                    model.unet.zero_grad()
                    attention_maps_erase = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=True)
                    self_attention_maps_erase = aggregate_attention(controller, 16, ["up", "down", "mid"], is_cross=False)

                    # attention loss
                    loss = attn_loss_func(attention_maps, attention_maps_erase, self_attention_maps, self_attention_maps_erase)
                    if loss != .0:
                        pbar.set_postfix({'loss': loss if isinstance(loss, float) else loss.item()})
                        text_embeddings_erase = update_context(context=text_embeddings_erase, loss=loss,
                                                               scale=scale, factor=np.sqrt(scale_range[i]))
                    del noise_prediction_text
                    torch.cuda.empty_cache()
                    controller.text_embeddings_erase = text_embeddings_erase.clone().detach().requires_grad_(False)
                iter -= 1

        # "uncond_embeddings_ is None" for real images, "uncond_embeddings_ is not None" for generated images.
        context_erase = controller.text_embeddings_erase
        controller.attention_store = {}
        controller.baseline = False
        contexts = (uncond_embeddings_, context_erase)
        latents = diffusion_step(model, controller, latents, contexts, t, guidance_scale, low_resource=LOW_RESOURCE)
        _latent_erase = latents

    if return_type == 'image':
        image = latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def update_context(context: torch.Tensor, loss: torch.Tensor, scale: int, factor: float) -> torch.Tensor:
    """
    Update the text embeddings according to the attention loss.

    :param context: text embeddings to be updated
    :param loss: ours loss
    :param factor: factor for update text embeddings.
    :return:
    """
    grad_cond = torch.autograd.grad(outputs=loss.requires_grad_(True), inputs=[context], retain_graph=False)[0]
    context = context - (scale * factor) * grad_cond
    return context

def run_and_display(ldm_stable, prompts, controller, latent=None, generator=None, uncond_embeddings=None, verbose=True):
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent,
                                        num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE,
                                        generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        view_images(images)
    return images, x_t

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_model(ckpt_path):
    ldm_stable = StableDiffusionPipeline.from_pretrained(ckpt_path).to("cuda")
    return ldm_stable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inversion', type=str, default='NT', help='NT (Null-text), NPI (Negative-prompt-inversion)')
    parser.add_argument('--seed', type=int, default=2, help='seed for generated image of stable diffusion')

    parser.add_argument('--cross_retain_steps', type=ast.literal_eval, default='[.2,]', help='perform the "wo" punish when step >= cross_wo_steps')  # .0 == τ=1.0, .1 == τ=0.9, .2 == τ=0.8
    parser.add_argument('--alpha', type=ast.literal_eval, default='[1.,]', help="punishment ratio")
    parser.add_argument('--max_step_to_erase', type=int, default=20, help='erase/suppress max step of diffusion model')
    parser.add_argument('--iter_each_step', type=int, default=10, help="the number of iteration for each step to update text embedding")
    parser.add_argument('--lambda_retain', type=float, default=1., help='lambda for cross attention retain loss')
    parser.add_argument('--lambda_erase', type=float, default=-.5, help='lambda for cross attention erase loss')
    parser.add_argument('--lambda_self_retain', type=float, default=1., help='lambda for self attention retain loss')
    parser.add_argument('--lambda_self_erase', type=float, default=-.5, help='lambda for self attention erase loss')
    parser.add_argument('--method', type=str, default='soft-weight', help='soft-weight, alpha, beta, delete, weight')

    parser.add_argument('--config_path', help='config path for stable diffusion v1-4 inference', type=str, required=False, default='configs/generate_sd.yaml')
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, required=False, default='ckpts/sd_model/diffuser/step19999')
    parser.add_argument("--theme", required=True, type=str)
    parser.add_argument('--type', type=str, default='object', choices=['object', 'style'], help='the type of remove')
    args = parser.parse_args()
    return args

## suppression for generated image
def main_gen(args, stable):
    """
    suppress content with EOT of text embeddings for generated images from stable diffusion model
    with given prompts.

    :param args:
    :param stable: stable diffusion model.
    :return: None.
    """

    seed = args.seed
    seed_everything(args.seed)
    for test_theme in theme_available:
        for object_class in class_available:
            prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."
            # print(prompt)
            n = len(stable.tokenizer.encode(prompt))
            os.makedirs(args.output_dir, exist_ok=True)

            if args.theme == test_theme or args.theme == object_class:
                if args.type == 'object':
                    token_indices_list = [[2]]
                else:
                    token_indices_list = [[i + 5 for i in range(len(test_theme.split('_')) + 1)]]

                for token_indices in token_indices_list:
                    for cross_retain_steps in args.cross_retain_steps:
                        for alpha in args.alpha:
                            g_cpu = torch.Generator().manual_seed(seed)
                            print(f'|----------Suppress EOT (Generated-Image): token_indices={token_indices}, alpha={alpha}, cross_retain_steps(1-tau)={cross_retain_steps}----------|')
                            controller = AttentionStore(token_indices, alpha, args.method, cross_retain_steps, n, args.iter_each_step, args.max_step_to_erase,
                                                        lambda_retain=args.lambda_retain, lambda_erase=args.lambda_erase, lambda_self_retain=args.lambda_self_retain, lambda_self_erase=args.lambda_self_erase)

                            x_sample, x_t = run_and_display(stable, [prompt], controller, latent=None, generator=g_cpu, uncond_embeddings=None, verbose=False)
                            view_images([x_sample[0]], save_name=f'{args.output_dir}/{test_theme}_{object_class}_seed{seed}')

if __name__=="__main__":
    args = parse_args()

    args.output_dir = f"eval_results/mu_results/seot/style50/{args.theme}/"
    wandb.init(project="quick-canvas-sd-generation", name=args.theme, config=args)
    print(f"Saving generated images to {args.output_dir}")

    stable = load_model(args.ckpt_path)
    main_gen(args, stable)