import torch
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import argparse
from dataset import setup_model, setup_forget_style_data

from constants.const import theme_available, class_available

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f'{word}_loss')
    plt.legend(loc="upper left")
    plt.title('Average loss in trainings', fontsize=20)
    plt.xlabel('Data point', fontsize=16)
    plt.ylabel('Loss value', fontsize=16)
    plt.savefig(path)
   
   
def generate_forget_style_mask(prompt, output_dir, forget_data_dir, remain_data_dir, c_guidance, batch_size, config_path, ckpt_path, device, lr=1e-5, image_size=512, num_timesteps=1000, threshold=0.5):
    # MODEL TRAINING SETUP
    model = setup_model(config_path, ckpt_path, device)
    forget_dl, remain_dl = setup_forget_style_data(forget_data_dir, remain_data_dir, batch_size, image_size)
    print(len(forget_dl))
    
    # set model to train
    model.train()
    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.diffusion_model.parameters(), lr=lr)

    gradients = {}
    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = 0

    # TRAINING CODE
    for epoch in range(1):
        with tqdm(total=len(forget_dl)) as t:
            for i, (_, _) in enumerate(forget_dl):
                optimizer.zero_grad()
                
                images, _ = next(iter(forget_dl))
                images = images.to(device)
                t = torch.randint(0, num_timesteps, (images.shape[0],), device=device).long()
                
                null_prompts = [""] * batch_size
                prompts = [prompt] * batch_size
                print(prompts)     
                
                forget_batch = {
                    "edited": images,
                    "edit": {"c_crossattn": prompts}
                }
                
                null_batch = {
                    "edited": images,
                    "edit": {"c_crossattn": prompts}
                } 
                
                forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
                null_input, null_emb = model.get_input(null_batch, model.first_stage_key)
                
                t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
                noise = torch.randn_like(forget_input, device=device)
                
                forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
                
                forget_out = model.apply_model(forget_noisy, t, forget_emb)
                null_out = model.apply_model(forget_noisy, t, null_emb)
                
                preds = (1 + c_guidance) * forget_out - c_guidance * null_out

                # print(images.shape, noise.shape, preds.shape)
                loss = - criteria(noise, preds)
                           
                loss.backward()
                optimizer.step()
                
                for name, param in model.model.diffusion_model.named_parameters():
                  if param.grad is not None:                    
                      gradient = param.grad.data.cpu()
                      gradients[name] += gradient

    for name, param in model.model.diffusion_model.named_parameters():
        gradients[name] = torch.abs(gradients[name])

    sorted_dict_positions = {}
    hard_dict = {}

    # Concatenate all tensors into a single tensor
    all_elements = torch.cat([tensor.flatten() for tensor in gradients.values()])

    # Calculate the threshold index for the top 10% elements
    threshold_index = int(len(all_elements) * threshold)

    # Calculate positions of all elements
    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in gradients.items():
        num_elements = tensor.numel()
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index: start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        threshold_tensor = torch.zeros_like(tensor_ranks)
        threshold_tensor[tensor_ranks < threshold_index] = 1
        threshold_tensor = threshold_tensor.reshape(tensor.shape)
        hard_dict[key] = threshold_tensor
        start_index += num_elements

    torch.save(hard_dict, os.path.join(output_dir, f'{threshold}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'Train', description = 'train a stable diffusion model from scratch')
    
    parser.add_argument('--c_guidance', help='guidance of start image used to train', type=float, required=False, default=7.5)
    parser.add_argument('--batch_size', help='batch_size used to train', type=int, required=False, default=4)
    parser.add_argument('--ckpt_path', help='ckpt path for stable diffusion v1-4', type=str, default="../main_sd_image_editing/ckpts/sd_model/compvis/style50/step6999.ckpt")
    parser.add_argument('--config_path', type=str, required=False, default='configs/train_salun.yaml')
    parser.add_argument('--num_timesteps', help='ddim steps of inference used to train', type=int, required=False, default=1000)
    parser.add_argument('--theme', help='prompt used to train', type=str, required=True, choices=theme_available + class_available)
    parser.add_argument('--output_dir', help='output dir for mask', type=str, required=True)

    parser.add_argument('--threshold', help='threshold for mask', type=float, required=False, default=0.4)

    parser.add_argument('--forget_data_dir', help='forget data dir', type=str, required=False, default='data')
    parser.add_argument('--remain_data_dir', help='remain data dir', type=str, required=False, default='data/Seed_Images/')
    args = parser.parse_args()
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    prompt = f"An image in {args.theme} Style."
    output_dir = os.path.join(args.output_dir, args.theme)
    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(os.path.join(output_dir, f'{args.threshold}.pt')):
        print(f"Mask for threshold {args.threshold} already exists. Skipping.")
        exit(0)

    forget_data_dir = os.path.join(args.forget_data_dir, args.theme)

    generate_forget_style_mask(prompt, output_dir, forget_data_dir, args.remain_data_dir, args.c_guidance, args.batch_size, args.config_path, args.ckpt_path, device, threshold=args.threshold)

    # python train-scripts/generate_mask.py --ckpt_path "models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt"