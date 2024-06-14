import argparse  
import gc  
import os  
from pathlib import Path  
import math  
import yaml
import wandb 
import torch 
import torch.nn as nn  
from typing import Literal, Optional, List  
from pydantic import BaseModel  
from transformers import CLIPTextModel, CLIPTokenizer  
from diffusers import (  
    UNet2DConditionModel,
    StableDiffusionPipeline,
    AltDiffusionPipeline,  
    DiffusionPipeline,  
)  
import safetensors
from pytorch_lightning import seed_everything
from safetensors.torch import load_file, save_file

import sys
sys.path.append(".")
from constants.const import theme_available, class_available

class GenerationConfig(BaseModel):
    prompts: list[str] = []
    negative_prompt: str = "bad anatomy,watermark,extra digit,signature,worst quality,jpeg artifacts,normal quality,low quality,long neck,lowres,error,blurry,missing fingers,fewer digits,missing arms,text,cropped,Humpbacked,bad hands,username"
    unconditional_prompt: str = ""
    width: int = 512
    height: int = 512
    num_inference_steps: int = 100
    guidance_scale: float = 7.5
    seed: int = 2024
    generate_num: int = 1

    save_path: str = "/localscratch/chongyu/quick-canvas-benchmark/SPM/output/img_{}.png"  # can be a template, e.g. "path/to/img_{}.png",
    # then the generated images will be saved as "path/to/img_0.png", "path/to/img_1.png", ...

    def dict(self):
        results = {}
        for attr in vars(self):
            if not attr.startswith("_"):
                results[attr] = getattr(self, attr)
        return results
    
    @staticmethod
    def fix_format(cfg):
        for k, v in cfg.items():
            if isinstance(v, list):
                cfg[k] = v[0]
            elif isinstance(v, torch.Tensor):
                cfg[k] = v.item()

def load_config_from_yaml(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return GenerationConfig(**cfg)

def parse_precision(precision: str) -> torch.dtype:
    if precision == "fp32" or precision == "float32":
        return torch.float32
    elif precision == "fp16" or precision == "float16":
        return torch.float16
    elif precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16

    raise ValueError(f"Invalid precision type: {precision}")

def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids

def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]

def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
    return_tokens: bool = False,
):
    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    if return_tokens:
        return text_embeddings, torch.unique(text_tokens, dim=1)
    return text_embeddings


TOKENIZER_V1_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
TOKENIZER_V2_MODEL_NAME = "stabilityai/stable-diffusion-2-1"

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]
DIFFUSERS_CACHE_DIR = ".cache/"  # if you want to change the cache dir, change this
LOCAL_ONLY = False  # if you want to use only local files, change this


def load_checkpoint_model(
    checkpoint_path: str,
    v2: bool = False,
    clip_skip: Optional[int] = None,
    weight_dtype: torch.dtype = torch.float32,
    device = "cuda",
) -> tuple[CLIPTokenizer, CLIPTextModel, UNet2DConditionModel, DiffusionPipeline]:
    print(f"Loading checkpoint from {checkpoint_path}")
    if checkpoint_path == "BAAI/AltDiffusion":
        pipe = AltDiffusionPipeline.from_pretrained(
            "BAAI/AltDiffusion", 
            upcast_attention=True if v2 else False,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            local_files_only=LOCAL_ONLY,
        ).to(device)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint_path,
            upcast_attention=True if v2 else False,
            torch_dtype=weight_dtype,
            cache_dir=DIFFUSERS_CACHE_DIR,
            local_files_only=LOCAL_ONLY,
        ).to(device)

    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    if clip_skip is not None:
        if v2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    return tokenizer, text_encoder, unet, pipe


class SPMLayer(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        spm_name,
        org_module: nn.Module,
        multiplier=1.0,
        dim=4,
        alpha=1,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.spm_name = spm_name
        self.dim = dim

        if org_module.__class__.__name__ == "Linear":
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, dim, bias=False)
            self.lora_up = nn.Linear(dim, out_dim, bias=False)

        elif org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.dim = min(self.dim, in_dim, out_dim)
            if self.dim != dim:
                print(f"{spm_name} dim (rank) is changed to: {self.dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )
        

class SPMNetwork(nn.Module):
    UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
        "Transformer2DModel",
    ]
    UNET_TARGET_REPLACE_MODULE_CONV = [
        "ResnetBlock2D",
        "Downsample2D",
        "Upsample2D",
    ]

    SPM_PREFIX_UNET = "lora_unet"   # aligning with SD webui usage
    DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        module = SPMLayer,
        module_kwargs = None,
    ) -> None:
        super().__init__()

        self.multiplier = multiplier
        self.dim = rank
        self.alpha = alpha

        self.module = module
        self.module_kwargs = module_kwargs or {}

        # unet spm
        self.unet_spm_layers = self.create_modules(
            SPMNetwork.SPM_PREFIX_UNET,
            unet,
            SPMNetwork.DEFAULT_TARGET_REPLACE,
            self.dim,
            self.multiplier,
        )
        print(f"Create SPM for U-Net: {len(self.unet_spm_layers)} modules.")

        spm_names = set()
        for spm_layer in self.unet_spm_layers:
            assert (
                spm_layer.spm_name not in spm_names
            ), f"duplicated SPM layer name: {spm_layer.spm_name}. {spm_names}"
            spm_names.add(spm_layer.spm_name)

        for spm_layer in self.unet_spm_layers:
            spm_layer.apply_to()
            self.add_module(
                spm_layer.spm_name,
                spm_layer,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
    ) -> list:
        spm_layers = []

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d"]:
                        spm_name = prefix + "." + name + "." + child_name
                        spm_name = spm_name.replace(".", "_")
                        print(f"{spm_name}")
                        spm_layer = self.module(
                            spm_name, child_module, multiplier, rank, self.alpha, **self.module_kwargs
                        )
                        spm_layers.append(spm_layer)

        return spm_layers

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        all_params = []

        if self.unet_spm_layers:
            params = []
            [params.extend(spm_layer.parameters()) for spm_layer in self.unet_spm_layers]
            param_data = {"params": params}
            if default_lr is not None:
                param_data["lr"] = default_lr
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def __enter__(self):
        for spm_layer in self.unet_spm_layers:
            spm_layer.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for spm_layer in self.unet_spm_layers:
            spm_layer.multiplier = 0


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        print(file_name)
        sd = load_file(file_name)
        metadata = load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def merge_lora_models(models, ratios, merge_dtype):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    for model, ratio in zip(models, ratios):
        print(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        print(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        print(f"merging...")
        for key in lora_sd.keys():
            if "alpha" in key:
                continue

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size()
                ), f"weights shape mismatch merging v1 and v2, different dims? / 重みのサイズが合いません。v1とv2、または次元数の異なるモデルはマージできません"
                merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)

    print("merged model")
    print(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

    # check all dims are same
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = True
    all_same_alphas = True
    for dims in dims_list:
        if dims != dims_list[0]:
            all_same_dims = False
            break
    for alphas in alphas_list:
        if alphas != alphas_list[0]:
            all_same_alphas = False
            break

    # build minimum metadata
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"

    return merged_sd


def merge_to_sd_model(text_encoder, unet, models, ratios, merge_dtype='cuda'):
    text_encoder.to(merge_dtype)
    unet.to(merge_dtype)

    # create module map
    name_to_module = {}
    for i, root_module in enumerate([text_encoder, unet]):
        if i == 0:
            prefix = 'lora_te'
            target_replace_modules = ['CLIPAttention', 'CLIPMLP']
        else:
            prefix = 'lora_unet'
            target_replace_modules = (
                ['Transformer2DModel'] + ['ResnetBlock2D', 'Downsample2D', 'Upsample2D']
            )

        for name, module in root_module.named_modules():
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ == "Linear" or child_module.__class__.__name__ == "Conv2d":
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        name_to_module[lora_name] = child_module

    for model, ratio in zip(models, ratios):
        print(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)

        print(f"merging...")
        for key in lora_sd.keys():
            if "lora_down" in key:
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                # find original module for this layer
                module_name = ".".join(key.split(".")[:-2])  # remove trailing ".lora_down.weight"
                if module_name not in name_to_module:
                    print(f"no module found for weight: {key}")
                    continue
                module = name_to_module[module_name]
                # print(f"apply {key} to {module}")

                down_weight = lora_sd[key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                # W <- W + U * D
                weight = module.weight
                if len(weight.size()) == 2:
                    # linear
                    if len(up_weight.size()) == 4:  # use linear projection mismatch
                        up_weight = up_weight.squeeze(3).squeeze(2)
                        down_weight = down_weight.squeeze(3).squeeze(2)
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # print(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                module.weight = torch.nn.Parameter(weight)


DEVICE_CUDA = torch.device("cuda:0")
UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"
MATCHING_METRICS = Literal[
    "clipcos",
    "clipcos_tokenuni",
    "tokenuni",
]

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def calculate_matching_score(
        prompt_tokens,
        prompt_embeds, 
        erased_prompt_tokens, 
        erased_prompt_embeds, 
        matching_metric: MATCHING_METRICS,
        special_token_ids: set[int],
        weight_dtype: torch.dtype = torch.float32,
    ):
    scores = []
    if "clipcos" in matching_metric:
        clipcos = torch.cosine_similarity(
                    prompt_embeds.flatten(1, 2), 
                    erased_prompt_embeds.flatten(1, 2), 
                    dim=-1).cpu()
        scores.append(clipcos)
    if "tokenuni" in matching_metric:
        prompt_set = set(prompt_tokens[0].tolist()) - special_token_ids
        tokenuni = []
        for ep in erased_prompt_tokens:
            ep_set = set(ep.tolist()) - special_token_ids
            tokenuni.append(len(prompt_set.intersection(ep_set)) / len(ep_set))
        scores.append(torch.tensor(tokenuni).to("cpu", dtype=weight_dtype))
    return torch.max(torch.stack(scores), dim=0)[0]

def infer_with_spm(
        spm_paths: list[str],
        config: GenerationConfig,
        matching_metric: MATCHING_METRICS,
        assigned_multipliers: list[float] = None,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
        seed: int = 188,
        theme: str = "Dogs"
    ):

    seed_everything(seed)

    spm_model_paths = [lp / f"{lp.name}_last.safetensors" if lp.is_dir() else lp for lp in spm_paths]
    weight_dtype = parse_precision(precision)
    
    # load the pretrained SD
    tokenizer, text_encoder, unet, pipe = load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    # load the SPM modules
    spms, metadatas = zip(*[
        load_state_dict(spm_model_path, weight_dtype) for spm_model_path in spm_model_paths
    ])
    # check if SPMs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    # get the erased concept
    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    print(f"Erased prompts: {erased_prompts}")

    erased_prompts_flatten = [item for sublist in erased_prompts for item in sublist]
    erased_prompt_embeds, erased_prompt_tokens = encode_prompts(
        tokenizer, text_encoder, erased_prompts_flatten, return_tokens=True
        )

    network = SPMNetwork(
        unet,
        rank=int(float(metadatas[0]["rank"])),
        alpha=float(metadatas[0]["alpha"]),
        module=SPMLayer,
    ).to(DEVICE_CUDA, dtype=weight_dtype)

    with torch.no_grad():
        for test_theme in theme_available:
            for object_class in class_available:
                prompt = f"A {object_class} image in {test_theme.replace('_', ' ')} style."

                prompt += config.unconditional_prompt
                print(f"Generating for prompt: {prompt}")
                prompt_embeds, prompt_tokens = encode_prompts(
                    tokenizer, text_encoder, [prompt], return_tokens=True
                    )
                if assigned_multipliers is not None:
                    multipliers = torch.tensor(assigned_multipliers).to("cpu", dtype=weight_dtype)
                    if assigned_multipliers == [0,0,0]:
                        matching_metric = "aazeros"
                    elif assigned_multipliers == [1,1,1]:
                        matching_metric = "zzone"
                else:
                    multipliers = calculate_matching_score(
                        prompt_tokens,
                        prompt_embeds, 
                        erased_prompt_tokens, 
                        erased_prompt_embeds, 
                        matching_metric=matching_metric,
                        special_token_ids=special_token_ids,
                        weight_dtype=weight_dtype
                        )
                    multipliers = torch.split(multipliers, erased_prompts_count)
                print(f"multipliers: {multipliers}")
                weighted_spm = dict.fromkeys(spms[0].keys())
                used_multipliers = []
                for spm, multiplier in zip(spms, multipliers):
                    max_multiplier = torch.max(multiplier)
                    for key, value in spm.items():
                        if weighted_spm[key] is None:
                            weighted_spm[key] = value * max_multiplier
                        else:
                            weighted_spm[key] += value * max_multiplier
                    used_multipliers.append(max_multiplier.item())
                network.load_state_dict(weighted_spm)
                with network:
                    image = pipe(
                        negative_prompt=config.negative_prompt,
                        width=config.width,
                        height=config.height,
                        num_inference_steps=config.num_inference_steps,
                        guidance_scale=config.guidance_scale,
                        generator=torch.cuda.manual_seed(seed),
                        num_images_per_prompt=config.generate_num,
                        prompt_embeds=prompt_embeds,
                    ).images[0]
                    
                image.save(os.path.join(args.output_dir, f"{test_theme}_{object_class}_seed{seed}.jpg"))


def main(args):
    spm_path = [Path(lp) for lp in args.spm_path]
    generation_config = load_config_from_yaml(args.config)

    infer_with_spm(
        spm_path,
        generation_config,
        args.matching_metric,
        assigned_multipliers=args.spm_multiplier,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
        seed=args.seed,
        theme=args.theme
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spm_multiplier",
        nargs="*",
        type=float,
        default=None,
        help="Assign multipliers for SPM model or set to `None` to use Facilitated Transport.",
    )
    parser.add_argument(
        "--matching_metric",
        type=str,
        default="clipcos_tokenuni",
        help="matching metric for prompt vs erased concept",
    )
    
    # model configs
    parser.add_argument(
        "--base_model",
        type=str,
        default="ckpts/sd_model/diffuser/step19999",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )

    # unlearn canvas
    parser.add_argument(
        "--seed", 
        type=int, 
        default=188, 
        help='seed for generated image of stable diffusion'
    )
    parser.add_argument(
        "--theme",
        required=True,
        type=str
    )

    args = parser.parse_args()
    args.output_dir = f"eval_results/mu_results/spm/style50/{args.theme}"
    os.makedirs(args.output_dir, exist_ok=True)

    args.config = f"../mu_semipermeable_membrane_spm/configs/{args.theme}/config.yaml"
    args.spm_path = [f"../mu_semipermeable_membrane_spm/output/{args.theme}/{args.theme}_last.safetensors"]
    wandb.init(project="quick-canvas-sd-generation", name=args.theme, config=args)

    main(args)