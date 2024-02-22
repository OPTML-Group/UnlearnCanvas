import argparse
import ast
import copy
import operator
import os
from functools import reduce

import torch
from diffusers import StableDiffusionPipeline
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from constants.const import theme_available, class_available

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./cache")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="./cache")

## get arguments for our script
with_to_k = False
with_augs = True
train_func = "train_closed_form"

### load model
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def edit_model(ldm_stable, old_text_, new_text_, retain_text_,
               layers_to_edit=None, lamb=0.1, erase_scale=0.1,
               preserve_scale=0.1, with_to_k=True, technique='tensor'):
    ### collect all the cross attns modules
    sub_nets = ldm_stable.unet.named_children()
    ca_layers = []
    for net in sub_nets:
        if 'up' in net[0] or 'down' in net[0]:
            for block in net[1]:
                if 'Cross' in block.__class__.__name__:
                    for attn in block.attentions:
                        for transformer in attn.transformer_blocks:
                            ca_layers.append(transformer.attn2)
        if 'mid' in net[0]:
            for attn in net[1].attentions:
                for transformer in attn.transformer_blocks:
                    ca_layers.append(transformer.attn2)

    ### get the value and key modules
    projection_matrices = [l.to_v for l in ca_layers]
    og_matrices = [l.to_v for l in ca_layers]
    if with_to_k:
        projection_matrices = projection_matrices + [l.to_k for l in ca_layers]
        og_matrices = og_matrices + [l.to_k for l in ca_layers]

    ## reset the parameters
    num_ca_clip_layers = len(ca_layers)
    for idx_, l in enumerate(ca_layers):
        l.to_v = copy.deepcopy(og_matrices[idx_])
        projection_matrices[idx_] = l.to_v
        if with_to_k:
            l.to_k = copy.deepcopy(og_matrices[num_ca_clip_layers + idx_])
            projection_matrices[num_ca_clip_layers + idx_] = l.to_k

    ### check the layers to edit (by default it is None; one can specify)
    layers_to_edit = ast.literal_eval(layers_to_edit) if type(layers_to_edit) == str else layers_to_edit
    lamb = ast.literal_eval(lamb) if type(lamb) == str else lamb

    ### Format the edits
    old_texts = []
    new_texts = []
    for old_text, new_text in zip(old_text_, new_text_):
        old_texts.append(old_text)
        n_t = new_text
        if n_t == '':
            n_t = ' '
        new_texts.append(n_t)
    if retain_text_ is None:
        ret_texts = ['']
    else:
        ret_texts = retain_text_

    print(old_texts, new_texts)
    ######################## START ERASING ###################################
    for layer_num in tqdm(range(len(projection_matrices))):
        if (layers_to_edit is not None) and (layer_num not in layers_to_edit):
            continue

        with torch.autocast("cuda"):
            #### prepare input k* and v*
            with torch.no_grad():
                # mat1 = \lambda W + \sum{v k^T}
                mat1 = lamb * projection_matrices[layer_num].weight

                # mat2 = \lambda I + \sum{k k^T}
                mat2 = lamb * torch.eye(projection_matrices[layer_num].weight.shape[1],
                                        device=projection_matrices[layer_num].weight.device)

                for cnt, t in enumerate(zip(old_texts, new_texts)):
                    old_text = t[0]
                    new_text = t[1]
                    texts = [old_text, new_text]
                    text_input = ldm_stable.tokenizer(
                        texts,
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]

                    final_token_idx = text_input.attention_mask[0].sum().item() - 2
                    final_token_idx_new = text_input.attention_mask[1].sum().item() - 2
                    farthest = max([final_token_idx_new, final_token_idx])

                    old_emb = text_embeddings[0]
                    old_emb = old_emb[final_token_idx:len(old_emb) - max(0, farthest - final_token_idx)]
                    new_emb = text_embeddings[1]
                    new_emb = new_emb[final_token_idx_new:len(new_emb) - max(0, farthest - final_token_idx_new)]

                    context = old_emb.detach()

                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            if technique == 'tensor':
                                o_embs = layer(old_emb).detach()
                                u = o_embs
                                u = u / u.norm()

                                new_embs = layer(new_emb).detach()
                                new_emb_proj = (u * new_embs).sum()

                                target = new_embs - (new_emb_proj) * u
                                values.append(target.detach())
                            elif technique == 'replace':
                                values.append(layer(new_emb).detach())
                            else:
                                values.append(layer(new_emb).detach())
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += erase_scale * for_mat1
                    mat2 += erase_scale * for_mat2

                for old_text, new_text in zip(ret_texts, ret_texts):
                    text_input = ldm_stable.tokenizer(
                        [old_text, new_text],
                        padding="max_length",
                        max_length=ldm_stable.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    )
                    text_embeddings = ldm_stable.text_encoder(text_input.input_ids.to(ldm_stable.device))[0]
                    old_emb, new_emb = text_embeddings
                    context = old_emb.detach()
                    values = []
                    with torch.no_grad():
                        for layer in projection_matrices:
                            values.append(layer(new_emb[:]).detach())
                    context_vector = context.reshape(context.shape[0], context.shape[1], 1)
                    context_vector_T = context.reshape(context.shape[0], 1, context.shape[1])
                    value_vector = values[layer_num].reshape(values[layer_num].shape[0], values[layer_num].shape[1], 1)
                    for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
                    for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
                    mat1 += preserve_scale * for_mat1
                    mat2 += preserve_scale * for_mat2
                    # update projection matrix
                projection_matrices[layer_num].weight = torch.nn.Parameter(mat1 @ torch.inverse(mat2))

    print(f'Current model status: Edited "{str(old_text_)}" into "{str(new_texts)}" and Retained "{str(retain_text_)}"')
    return ldm_stable


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='TrainUSD',
        description='Finetuning stable diffusion to debias the concepts')
    parser.add_argument('--guided_concepts', help='Concepts to guide the erased concepts', type=str,
                        default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve', type=str, default=None)
    parser.add_argument('--technique', help='technique to erase (either replace or tensor)', type=str, required=False,
                        default='replace')
    parser.add_argument('--ckpt', help='path to checkpoint', type=str,
                        default='../main_sd_image_editing/ckpts/sd_model/diffuser/style50/step19999')
    parser.add_argument('--theme', help='theme to forget', type=str, required=True,
                        choices=theme_available + class_available)
    parser.add_argument('--preserve_scale', help='scale to preserve concepts', type=float, required=False, default=None)
    parser.add_argument('--preserve_number', help='number of preserve concepts', type=int, required=False, default=None)
    parser.add_argument('--erase_scale', help='scale to erase concepts', type=float, required=False, default=1)
    parser.add_argument('--lamb', help='lambda for the loss', type=float, required=False, default=0.5)
    parser.add_argument('--add_prompts', help='option to add additional prompts', action="store_true", required=False,
                        default=False)
    parser.add_argument('--output_dir', type=str, default="results/style50/")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.theme)
    technique = args.technique
    device = 'cuda'
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    add_prompts = args.add_prompts
    guided_concepts = args.guided_concepts
    preserve_concepts = args.preserve_concepts
    preserve_number = args.preserve_number
    concepts = [args.theme]

    old_texts = []

    additional_prompts = []
    if args.theme in theme_available:
        additional_prompts.append('image in {concept} Style')
        additional_prompts.append('art by {concept}')
        additional_prompts.append('artwork by {concept}')
        additional_prompts.append('picture by {concept}')
        additional_prompts.append('style of {concept}')
    else:  # args.theme in class_available
        additional_prompts.append('image of {concept}')
        additional_prompts.append('photo of {concept}')
        additional_prompts.append('portrait of {concept}')
        additional_prompts.append('picture of {concept}')
        additional_prompts.append('painting of {concept}')
    if not add_prompts:
        additional_prompts = []
    concepts_ = []
    for concept in concepts:
        old_texts.append(f'{concept}')
        for prompt in additional_prompts:
            old_texts.append(prompt.format(concept=concept))
        length = 1 + len(additional_prompts)
        concepts_.extend([concept] * length)

    if guided_concepts is None:
        new_texts = [' ' for _ in old_texts]
    else:
        guided_concepts = [con.strip() for con in guided_concepts.split(',')]
        if len(guided_concepts) == 1:
            new_texts = [guided_concepts[0] for _ in old_texts]
        else:
            new_texts = [[con] * length for con in guided_concepts]
            new_texts = reduce(operator.concat, new_texts)
    assert len(new_texts) == len(old_texts)

    retain_texts = [""]

    for theme in theme_available:
        if args.theme == theme:
            continue
        if theme == "Seed_Images":
            theme = "Photo"
        for concept in class_available:
            if concept == args.theme:
                continue
            retain_texts.append(f'A {concept} image in {theme} style')

    if preserve_scale is None:
        preserve_scale = max(0.1, 1 / len(retain_texts))
    ldm_stable = StableDiffusionPipeline.from_pretrained(args.ckpt, torch_dtype=torch.float16).to(device)
    print("Old texts: ", old_texts)
    print("New texts: ", new_texts)
    ldm_stable = edit_model(ldm_stable=ldm_stable, old_text_=old_texts, new_text_=new_texts,
                            retain_text_=retain_texts, lamb=args.lamb, erase_scale=erase_scale,
                            preserve_scale=preserve_scale,
                            technique=technique)

    torch.save(ldm_stable.unet.state_dict(), output_path)
