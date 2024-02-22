import glob
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import wandb
from einops import rearrange
import sys
sys.path.append('.')
from stable_diffusion.ldm.models.diffusion.ddim import DDIMSampler
from stable_diffusion.ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    token_weights = sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    del sd["cond_stage_model.transformer.text_model.embeddings.token_embedding.weight"]
    m, u = model.load_state_dict(sd, strict=False)
    model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[
        :token_weights.shape[0]] = token_weights
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model


def initialize(config, ckpt, delta_ckpt, seed=42):
    "initialize a model and sampler given checkpoing path"
    if delta_ckpt is not None:
        if len(glob.glob(os.path.join(delta_ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            config = sorted(glob.glob(os.path.join(
                delta_ckpt.split('checkpoints')[0], "configs/*.yaml")))[-1]
    else:
        if len(glob.glob(
                os.path.join(ckpt.split('checkpoints')[0], "configs/*.yaml"))) > 0:
            config = sorted(
                glob.glob(os.path.join(ckpt.split('checkpoints')[0], "configs/*.yaml")))[
                -1]

    seed_everything(seed)
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    if delta_ckpt is not None:
        delta_st = torch.load(delta_ckpt)
        embed = None
        if 'embed' in delta_st:
            delta_st['state_dict'] = {}
            delta_st['state_dict']['embed'] = delta_st['embed']
        if 'embed' in delta_st['state_dict']:
            embed = delta_st['state_dict']['embed'].reshape(-1, 768)
            del delta_st['state_dict']['embed']
            print(embed.shape)
        model.load_state_dict(delta_st['state_dict'], strict=False)
        if embed is not None:
            print(f"restoring embedding. Embedding shape: {embed.shape[0]}")
            model.cond_stage_model.transformer.text_model.embeddings.token_embedding.weight.data[
                -embed.shape[0]:] = embed

    device = torch.device('cuda')
    model = model.to(device)
    sampler = DDIMSampler(model)
    return model, sampler, device


def sample(data, model, sampler, outpath, ddim_steps=200, n_samples=10, ddim_eta=1.0,
           n_iter=1, scale=6, batch_size=10, shape=(4, 64, 64),
           fixed_code=False, device=None, skip_save=False, skip_grid=True,
           metadata=True, base_count=0, n_rows=5, wandb_log=False, ckptname='base', rank=None):
    """
        decoupled image sampling function, including saving, visualizing and wandb logging
    """
    batch_size = n_samples
    sample_path = os.path.join(outpath, "samples")
    if not Path(sample_path).exists():
        Path(sample_path).mkdir()

    if metadata:
        images_path = []
        captions = []
    start_code = None
    if fixed_code:
        start_code = torch.randn([batch_size, ] + list(shape), device=device)
    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in tqdm(data, desc="data"):
                    all_samples = list()
                    for n in trange(n_iter, desc="Sampling"):
                        print(prompts[0])
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(
                                batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=batch_size,
                                                         shape=list(shape),
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        # print(samples_ddim.size())
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp(
                            (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu()

                        if not skip_save:
                            for x_sample, caption in zip(x_samples_ddim, prompts):
                                x_sample = 255. * \
                                    rearrange(x_sample.cpu().numpy(),
                                              'c h w -> h w c')
                                img = Image.fromarray(
                                    x_sample.astype(np.uint8))
                                img.save(os.path.join(
                                    sample_path, f"{base_count:05}.png"))
                                if metadata:
                                    images_path.append(os.path.join(
                                        sample_path, f"{base_count:05}.png"))
                                    captions.append(caption)
                                base_count += 1

                        if not skip_grid:
                            all_samples.append(x_samples_ddim)

                    if not skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * \
                            rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))

                        img = img.convert('RGB')
                        prompt_name = "".join(
                            ch for ch in prompts[0] if ch.isalpha() or ch.isspace())
                        prompt_name = prompt_name.replace(" ", "-")
                        file_prompt_name = prompt_name[:60]
                        img.save(os.path.join(outpath,
                                              f'{file_prompt_name}_{scale}_{ddim_steps}_{ddim_eta}.jpg'),
                                 quality=70)
                        if wandb_log:
                            out = BytesIO()
                            img.save(out, format='jpeg', quality=70)
                            img = Image.open(out)
                            wandb.log({f'{file_prompt_name}_{scale}_{ddim_steps}_{ddim_eta}.jpg': [
                                wandb.Image(img, caption=ckptname)]})

                            out.close()

    image_txt_path = ''
    caption_txt_path = ''
    if metadata:
        if rank is None:
            image_txt_path = os.path.join(outpath, 'images.txt')
            caption_txt_path = os.path.join(outpath, 'caption.txt')
        else:
            image_txt_path = os.path.join(outpath, f'images{rank}.txt')
            caption_txt_path = os.path.join(outpath, f'caption{rank}.txt')
        with open(image_txt_path, 'w') as f:
            for i in images_path:
                f.write(i + '\n')
        with open(caption_txt_path, 'w') as f:
            for i in captions:
                f.write(i + '\n')
    print('++++++++++++++++++++++++++++++++++++')
    print('+ Generation Finished ! ++++++++++++')
    print('++++++++++++++++++++++++++++++++++++')
    return image_txt_path, caption_txt_path


def sample_images(data, rank, config, ckpt, delta_ckpt, outpath, base_count, ddim_steps, n_samples):
    torch.cuda.set_device(rank)
    model, sampler, device = initialize(config, ckpt, delta_ckpt)
    return sample(data, model, sampler, outpath, ddim_steps, n_samples, base_count=base_count, rank=rank)


def distributed_sample_images(data, ranks, config, ckpt, delta_ckpt, outpath, ddim_steps=200, n_samples=10):
    """
        data        : list of batch prompts (2-dim list)
        ranks       : list of available GPU-cards
        config      : the config file to load model
        ckpt        : the checkpoint path to model
        delta_ckpt  : the checkpoint path to delta model
        outpath     : the root folder to save images
        ddim_steps  : the ddim steps in generation
    """
    process_stack = []
    count = 0
    size = int(np.ceil(len(data) / len(ranks)))
    for i, local_rank in enumerate(ranks):
        cur_data = data[i*size:(i+1)*size]
        base_count = i*size * len(data[0])
        process = mp.Process(target=sample_images,
                             args=(cur_data, local_rank, config, ckpt, delta_ckpt, outpath, base_count, ddim_steps, n_samples))
        process.start()
        process_stack.append(process)
        count += 1
        # wait for each process running

    for process in process_stack:
        process.join()
    # merge metadata
    images_path = []
    captions = []
    for local_rank in ranks:
        cur_image_txt_path = os.path.join(outpath, f'images{local_rank}.txt')
        cur_caption_txt_path = os.path.join(
            outpath, f'caption{local_rank}.txt')
        with open(cur_image_txt_path, 'r') as f:
            images_path += f.read().splitlines()
        with open(cur_caption_txt_path, 'r') as f:
            captions += f.read().splitlines()
        os.remove(cur_image_txt_path)
        os.remove(cur_caption_txt_path)

    image_txt_path = os.path.join(outpath, 'images.txt')
    caption_txt_path = os.path.join(outpath, 'caption.txt')
    with open(image_txt_path, 'w') as f:
        for i in images_path:
            f.write(i + '\n')
    with open(caption_txt_path, 'w') as f:
        for i in captions:
            f.write(i + '\n')


def safe_dir(dir):
    if not dir.exists():
        dir.mkdir()
    return dir
