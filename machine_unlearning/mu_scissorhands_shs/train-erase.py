# ref 
# - https://github.com/JingWu321/Scissorhands_ex/tree/master/SD
import os
import sys

import torch
from tqdm import tqdm

sys.path.append('.')
import argparse
from dataset import setup_model, setup_forget_style_data
from constants.const import theme_available, class_available

import gc
import numpy as np
from timm.utils import AverageMeter
from timm.models.layers import trunc_normal_

import quadprog
import copy
import wandb

# projection
# https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().contiguous().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]  # task mums
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0] # get the optimal solution of v~
    x = np.dot(v, memories_np) + gradient_np  # g~ = v*GT +g
    new_grad = torch.Tensor(x).view(-1)
    return new_grad


def create_dense_mask(net, device, value=1):
    for param in net.parameters():
        param.data[param.data == param.data] = value
    net.to(device)
    return net


def snip(model, dataloader, sparsity, prune_num, device):
    criterion = torch.nn.MSELoss()

    # compute grads
    grads = [torch.zeros_like(p) for p in model.model.diffusion_model.parameters()]
    for ii in range(prune_num):

        forget_images, forget_prompts = next(iter(dataloader))
        # tuple -> list
        forget_prompts = list(forget_prompts)

        # forget stage
        forget_batch = {
            "edited": forget_images,
            "edit": {"c_crossattn": forget_prompts}
        }
        loss = model.shared_step(forget_batch)[0]
        model.model.diffusion_model.zero_grad()
        loss.backward()

        with torch.no_grad():
            j = 0
            for n, param in  model.model.diffusion_model.named_parameters():
                if (param.grad is not None):
                    grads[j] += (param.grad.data).abs()
                j += 1
            torch.cuda.empty_cache()
            gc.collect()

    # compute saliences to get the threshold
    weights = [p for p in model.model.diffusion_model.parameters()]
    mask_ = create_dense_mask(copy.deepcopy(model.model.diffusion_model), device, value=1)
    with torch.no_grad():
        abs_saliences = [(grad * weight).abs() for weight, grad in zip(weights, grads)]
        saliences = [saliences.view(-1).cpu() for saliences in abs_saliences]
        saliences = torch.cat(saliences)
        threshold = float(saliences.kthvalue(int(sparsity * saliences.shape[0]))[0]) # k-th smallest value

        # get mask to prune the weights
        for j, param in enumerate(mask_.parameters()):
            indx = (abs_saliences[j] > threshold) # prune for forget data
            param.data[indx] = 0

        # update the weights of the original network with the mask
        for (n, param), (m_param) in zip(model.model.diffusion_model.named_parameters(), mask_.parameters()):
            if ("attn2" in n):
                mask = torch.empty(param.data.shape, device=device)
                if ('weight' in n):
                    re_init_param = trunc_normal_(mask, std=.02)
                elif ('bias' in n):
                    re_init_param = torch.nn.init.zeros_(mask)
                param.data = param.data * m_param.data + re_init_param.data * (1 - m_param.data)

    return model


def style_removal(forget_data_dir, remain_data_dir, output_dir, config_path, ckpt_path, 
                  sparsity, project, memory_num, prune_num,
                  train_method, alpha=0.1, batch_size=4, epochs=5, lr=1e-5, device="cuda:0",
                  image_size=512):
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

    # prune via snip
    model = snip(model, forget_dl, sparsity, prune_num, device)
    condition = lambda n: ('attn2' in n)
    if project:
        proxy_model = copy.deepcopy(model).to(device)
        proxy_model.eval()
        g_o = []
        for ii in range(memory_num):

            forget_images, forget_prompts = next(iter(forget_dl))
            # tuple -> list
            forget_prompts = list(forget_prompts)

            # forget stage
            forget_batch = {
                "edited": forget_images,
                "edit": {"c_crossattn": forget_prompts}
            }
            loss = -proxy_model.shared_step(forget_batch)[0]

            loss.backward()
            grad_o = []
            for n, param in proxy_model.model.diffusion_model.named_parameters():
                if param.grad is not None:
                    if condition(n):
                        grad_o.append(param.grad.detach().view(-1))
            g_o.append(torch.cat(grad_o))
            torch.cuda.empty_cache()
            gc.collect()
        g_o = torch.stack(g_o, dim=1)
    
    # set model to train
    model.train()
    losses = []
    optimizer = torch.optim.Adam(parameters, lr=lr)
    criteria = torch.nn.MSELoss()

    # TRAINING CODE
    for epoch in range(epochs):
        with tqdm(total=len(forget_dl)) as time:
            for i, _ in enumerate(forget_dl):
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

                forget_loss = criteria(forget_out, pseudo_out)

                # total loss
                loss = alpha * forget_loss + remain_loss
                loss.backward()
                losses.append(loss.item() / batch_size)

                if project and (i % 10 == 0):
                    # get the gradient w.r.t the pruned model
                    grad_f = []
                    for n, param in model.model.diffusion_model.named_parameters():
                        # if (param.grad is not None) and condition(n):
                        if param.grad is not None:
                            if condition(n):
                                grad_f.append(param.grad)
                    g_f = torch.cat(list(map(lambda g: g.detach().view(-1), grad_f)))


                    # compute the dot product of the gradients
                    dotg = torch.mm(g_f.unsqueeze(0), g_o)
                    if ((dotg < 0).sum() != 0):
                        grad_new = project2cone2(g_f.unsqueeze(0), g_o)
                        # overwrite the gradient
                        pointer = 0
                        for n, p in model.model.diffusion_model.named_parameters():
                            if param.grad is not None:
                                if condition(n):
                                    this_grad = grad_new[pointer:pointer + p.numel()].view(p.grad.data.size()).to(device)
                                    p.grad.data.copy_(this_grad)
                                    pointer += p.numel()

                optimizer.step()
                torch.cuda.empty_cache()
                gc.collect()
                time.set_description('Epoch %i' % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                time.update(1)

    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='SHS',
        description='Finetuning stable diffusion model to erase concepts using SHS')

    parser.add_argument('--forget_data_dir', help='forget data dir', type=str, required=False, default='data')
    parser.add_argument('--remain_data_dir', help='remain data dir', type=str, required=False, default='data/Seed_Images/')
    parser.add_argument('--theme', help='prompt used to train', type=str, required=True, choices=theme_available + class_available)
    parser.add_argument('--train_method', help='method of training', type=str, default="full",
                        choices=["noxattn", "selfattn", "xattn", "full", "notime", "xlayer", "selflayer"])
    parser.add_argument('--alpha', help='guidance of start image used to train', type=float, required=False, default=0.1)
    parser.add_argument('--epochs', help='epochs used to train', type=int, required=False, default=2)
    parser.add_argument('--config_path', type=str, required=False, default='configs/train_shs.yaml')
    parser.add_argument('--ckpt_path', type=str, required=False,
                        default='../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt')
    parser.add_argument('--output_dir', help='output dir for mask', type=str, default='results')
    parser.add_argument('--input_dir', help='output dir for mask', type=str, default=None)

    parser.add_argument('--sparsity', help='threshold for mask', type=float, default=0.99)
    parser.add_argument('--project', action='store_true', default=False)
    parser.add_argument('--memory_num', type=int, default=1)
    parser.add_argument('--prune_num', type=int, default=1)

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
                  sparsity=args.sparsity, project=args.project, memory_num=args.memory_num, prune_num=args.prune_num, device=device)

    torch.save({"state_dict": model.state_dict()}, os.path.join(output_dir, "sd.ckpt"))