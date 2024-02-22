# Selective Amnesia (SA) for Stable Diffusion
This is the official repository for Selective Amnesia for Stable Diffusion. The code of this project
is modifed from the official [Stable Diffusion](https://github.com/CompVis/stable-diffusion) repository.

# Requirements 
Install requirements using a `conda` environment:
```
conda env create -f environment.yaml
conda activate sa-sd
```

## Download SD v1.4 Checkpoint
We will use the SD v1.4 checkpoint (with EMA). You can either download it from the official HuggingFace [link](https://huggingface.co/CompVis/stable-diffusion-v-1-4-original) and move it to the root directory of this project, or alternatively
```
wget https://huggingface.co/CompVis/stable-diffusion-v-1-4-original
```

# Forgetting Training with SA
We consider forgetting the celebrity Brad Pitt in the following steps.

1. Generate Samples For FIM and GR.

    The prompts for generating the dataset for FIM/GR is given in `fim_prompts.txt`. This was generated automatically from GPT3.5. To generate the samples from the prompts using SD v1.4, run
    ```
    python scripts/txt2img_fim_prompts.py --ckpt sd-v1-4-full-ema.ckpt --from-file fim_prompts.txt --outdir fim_dataset
    ```
    The images will be stored in the folder `fim_dataset`.

2. Calculate the FIM (can take a long time, for precomputed FIM see [below](#checkpoints-and-pre-computed-fim).
    ```
    python main_fim.py -t --base configs/stable-diffusion/fim.yaml --gpus "0,1,2,3" --num_nodes 1 --finetune_from sd-v1-4-full-ema.ckpt --n_chunks 20
    ```
    where the `--gpus` flag should be set to all the GPU IDs that you intend to use. `--n_chunks` can be increased if you are running out of VRAM. This should produce multiple `fisher_dict_rank_[GPU RANK].pkl` files, one for each GPU. Run
    ```
    python combine_fisher_dict.py
    ```
    to combine them into a single dictionary file `full_fisher_dict.pkl`.

3. Generate the surrogate dataset $q(x|c_f)$ represented by images of "middle aged man":
    ```
    python scripts/txt2img_make_n_samples.py --outdir q_dist/middle_aged_man_dataset --prompt "a middle aged man" --n_samples 1000
    ```

4. Forgetting Training with SA
    ```
    python main_forget.py -t --base configs/stable-diffusion/forget_brad_pitt.yaml --gpus "0,1,2,3" --num_nodes 1 --finetune_from sd-v1-4-full-ema.ckpt
    ```
    The results and checkpoint are saved in `logs`.

    ## Edit Config File
    You can edit the config files in `configs/stable-diffusion`. Parameters that you should pay attention to have accompanying comments in the config file. Here are some notable ones:
    $\lambda$ and the layers to train can be modified in lines 18 and 19
    ```
    ...
    lmbda: 50 # change FIM weighting term here
    train_method: 'full' # choices: ['full', 'xattn', 'noxattn']
    ...
    ```
    $c_f$ and the path to the surrogate distribution can be modified in lines 85 and 86
    ```
    ...
    forget_prompt: brad pitt
    forget_dataset_path: ./q_dist/middle_aged_man_dataset
    ...
    ```
    The number of training epochs can be adjusted in line 133
    ```
    ...
    max_epochs: 200
    ...
    ```

# Checkpoints and Pre-computed FIM

We release the pretrained checkpoints and precomputed FIM [here](https://huggingface.co/ajrheng/selective-amnesia/tree/main). You may use the checkpoints with `scripts/txt2img.py` for image generation right away. 

The FIM is calculated for SD v1.4 and can be used for your own training for SA. Download into the base `sd` folder as follows
```
wget https://huggingface.co/ajrheng/selective-amnesia/blob/main/full_fisher_dict.pkl
```
and you can skip step 2 above. 

# Evaluation

## NudeNet 
First generate the images from the I2P dataset. 
```
python scripts/txt2img_i2p.py --ckpt path/to/trained/checkpoint --outdir path/to/outdir
```

Then run the NudeNet evaluation
```
python nudenet_evaluator.py path/to/outdir
```
The statistics of the relevant nudity concepts will be printed to screen.

## GIPHY Celebrity Detector
First generate the images to be evaluated. To replicate the Brad Pitt experiments, run
```
python scripts/txt2img_from_file.py --ckpt path/to/trained/checkpoint --outdir path/to/outdir --from-file brad_pitt_prompts.txt 
```

Next, clone the [GIPHY Celebrity Detector](https://github.com/Giphy/celeb-detection-oss) and follow the [official installation instructions](https://github.com/Giphy/celeb-detection-oss/tree/master/examples).

You will need to replace two files in your cloned repo:
```
examples/inference.py
model_training/helpers/face_recognizer.py
```
with those provided in `modified-celeb-detection-oss`. 

Then run evaluation
```
python examples/inference.py --image_folder path/to/outdir --celebrity "brad_pitt"
```
The statistics will be printed to screen.