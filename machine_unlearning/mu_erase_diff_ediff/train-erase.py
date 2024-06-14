# ref
# - https://github.com/JingWu321/EraseDiff/tree/master/sd
import os
import sys

import torch
from tqdm import tqdm

sys.path.append('.')
import argparse
from dataset import setup_model, setup_forget_style_data
from constants.const import theme_available, class_available

import gc
from timm.utils import AverageMeter
import wandb

def get_param(net):
    new_param = []
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            new_param.append(param.clone())
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return new_param


def set_param(net, old_param):
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            param.copy_(old_param[j])
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return net


def style_removal(forget_data_dir, remain_data_dir, output_dir, config_path, ckpt_path, 
                  K_steps, train_method, alpha=0.1, batch_size=4, epochs=5, 
                  lr=5e-5, device="cuda:0", image_size=512):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    forget_dl, remain_dl = setup_forget_style_data(forget_data_dir, remain_data_dir, batch_size, image_size)
    print(len(forget_dl))

    # choose parameters to train based on train_method
    parameters = []
    for name, param in model.model.diffusion_model.named_parameters():
        # train all layers except x-attns and time_embed layers
        if train_method == 'noxattn':
            if name.startswith('out.') or 'attn2' in name or 'time_embed' in name:
                pass
            else:
                # print(name)
                parameters.append(param)
        # train only self attention layers
        if train_method == 'selfattn':
            if 'attn1' in name:
                # print(name)
                parameters.append(param)
        # train only x attention layers
        if train_method == 'xattn':
            if 'attn2' in name:
                # print(name)
                parameters.append(param)
        # train all layers
        if train_method == 'full':
            # print(name)
            parameters.append(param)
        # train all layers except time embed layers
        if train_method == 'notime':
            if not (name.startswith('out.') or 'time_embed' in name):
                # print(name)
                parameters.append(param)
        if train_method == 'xlayer':
            if 'attn2' in name:
                if 'output_blocks.6.' in name or 'output_blocks.8.' in name:
                    # print(name)
                    parameters.append(param)
        if train_method == 'selflayer':
            if 'attn1' in name:
                if 'input_blocks.4.' in name or 'input_blocks.7.' in name:
                    # print(name)
                    parameters.append(param)
    
    # set model to train
    model.train()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    # TRAINING CODE
    for epoch in range(epochs):

        model.train()
        with tqdm(total=len(forget_dl)) as time:

            param_i = get_param(model)

            for j in range(K_steps):
                unl_losses = AverageMeter()
                for i, _ in enumerate(forget_dl):
                    optimizer.zero_grad()

                    forget_images, forget_prompts = next(iter(forget_dl))
                    remain_images, remain_prompts = next(iter(remain_dl))

                    # tuple -> list
                    forget_prompts = list(forget_prompts)
                    remain_prompts = list(remain_prompts)

                    pseudo_prompts = remain_prompts

                    # forget stage
                    forget_batch = {
                        "edited": forget_images,
                        "edit": {"c_crossattn": forget_prompts}
                    }

                    pseudo_batch = {
                        "edited": forget_images,
                        "edit": {"c_crossattn": pseudo_prompts}
                    }

                    forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
                    pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)

                    t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=model.device).long()
                    noise = torch.randn_like(forget_input, device=model.device)

                    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                    pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                    forget_out = model.apply_model(forget_noisy, t, forget_emb)
                    pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                    forget_loss = criteria(forget_out, pseudo_out)
                    forget_loss.backward()

                    optimizer.step()
                    unl_losses.update(forget_loss)
                    torch.cuda.empty_cache()
                    gc.collect()

                model = set_param(model, param_i) # keep \theta_i for f and q
                used_num_samples = 0
                for i, _ in enumerate(forget_dl):
                    model.train()
                    optimizer.zero_grad()

                    forget_images, forget_prompts = next(iter(forget_dl))
                    remain_images, remain_prompts = next(iter(remain_dl))

                    # tuple -> list
                    forget_prompts = list(forget_prompts)
                    remain_prompts = list(remain_prompts)

                    pseudo_prompts = remain_prompts

                    # remain stage
                    remain_batch = {
                        "edited": remain_images,
                        "edit": {"c_crossattn": remain_prompts}
                    }
                    remain_loss = model.shared_step(remain_batch)[0]

                    # forget stage
                    forget_batch = {
                        "edited": forget_images,
                        "edit": {"c_crossattn": forget_prompts}
                    }

                    pseudo_batch = {
                        "edited": forget_images,
                        "edit": {"c_crossattn": pseudo_prompts}
                    }

                    forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
                    pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)

                    t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=model.device).long()
                    noise = torch.randn_like(forget_input, device=model.device)

                    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                    pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)

                    forget_out = model.apply_model(forget_noisy, t, forget_emb)
                    pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()

                    unlearn_loss = criteria(forget_out, pseudo_out)
                    q_loss = unlearn_loss - unl_losses.avg.detach()

                    total_loss = remain_loss + alpha * q_loss
                    total_loss.backward()
                    optimizer.step()

            time.set_description('Epoch %i' % epoch)
            time.set_postfix(loss=total_loss.item() / batch_size)
            time.update(1)

    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='EDiff',
        description='Finetuning stable diffusion model to erase concepts using EDiff')

    parser.add_argument('--forget_data_dir', help='forget data dir', type=str, required=False, default='data')
    parser.add_argument('--remain_data_dir', help='remain data dir', type=str, required=False, default='data/Seed_Images/')
    parser.add_argument('--theme', help='prompt used to train', type=str, required=True, choices=theme_available + class_available)
    parser.add_argument('--train_method', help='method of training', type=str, default="xattn",
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--alpha', help='guidance of start image used to train', type=float, required=False, default=0.1)
    parser.add_argument('--epochs', help='epochs used to train', type=int, required=False, default=5)
    parser.add_argument('--config_path', type=str, required=False, default='configs/train_ediff.yaml')
    parser.add_argument('--ckpt_path', type=str, required=False,
                        default='../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt')
    parser.add_argument('--output_dir', help='output dir for mask', type=str, default='results')
    parser.add_argument('--input_dir', help='output dir for mask', type=str, default=None)

    parser.add_argument('--K_steps', type=int, required=False, default=2)

    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = args.output_dir
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    forget_data_dir = os.path.join(args.forget_data_dir, args.theme)
    output_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(project="quick-canvas-machine-unlearning", name=args.theme, config=args)
    model = style_removal(forget_data_dir, args.remain_data_dir, output_dir=output_dir, config_path=args.config_path,
                  train_method=args.train_method, epochs=args.epochs, ckpt_path=args.ckpt_path, alpha=args.alpha,
                  K_steps=args.K_steps, device=device)

    torch.save({"state_dict": model.state_dict()}, os.path.join(output_dir, "sd.ckpt"))