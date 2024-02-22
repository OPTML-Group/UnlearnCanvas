# Fine-tune Stable Diffusion and Image Editing Model with UnlearnCanvas

This repository offers detailed instructions and source code for fine-tuning pre-trained models, specifically StableDiffusion and InstructPix2Pix, using the UnlearnCanvas dataset.

## Environment Setup

The experiments within this repository rely on the `unlearn_canvas` conda environment. Activate this environment prior to executing any scripts:

```bash
conda activate unlearn_canvas
```

## Fine-Tuning Process

To fine-tune models, users have the option to employ either the `compvis` or the `diffuser` codebase, each offering distinct pipelines tailored to varying requirements. The compvis pipeline allows for greater flexibility in modifications, whereas the diffuser pipeline provides enhanced integration and automation.

By default, fine-tuning is performed using the **StableDiffusion-v1.5** pretrained checkpoints. Configurations can be adjusted to utilize alternative checkpoints.

### Compvis Pipeline

#### Fine-Tuning Stable Diffusion and Image Editing Models

Ensure the dataset is located within `./data/unlearn_canvas/`. Configuration files may be modified to alter the dataset path.

* Fine-tune the stable diffusion model: Usually a learning rate of 1e-6 is recommended based on the experience, which has been set to default.
    ```bash
    python3 main.py --name sd_unlearn_canvas --base configs/train_sd.yaml --train --gpus 0,1,2,3,4,5,6,7
    ```
* Fine-tine the image editing (InstructPix2Pix) model: usually a learning rate of 1e-4 is recommended based on the experience, which has been set to default.
    ```bash
    python3 main.py --name ip2p_unlearn_canvas --base configs/train.yaml --train --gpus 0,1,2,3,4,5,6,7
    ```

The checkpoints are stored in the `logs` folder, specified by the `name` parameter you provided in the training commands. You can find the pretrained checkpoint with 50 styles in the [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`diffusion/compvis/style50/compvis.ckpt`).

### Sampling with Fine-Tuned Models

#### Sampling Fine-Tuned Stable Diffusion Model

To sample from the fine-tuned stable diffusion models, you can use the following scripts. 

* `sampling/stable_diffusion/sample_compvis_single.py`: this script aims to sample from the fine-tuned stable diffusion model using a particular single prompt as the input. This intends to provide a flexible way to let the users to try the model and play with different prompts. 
    ```bash
    python3 sampling/stable_diffusion/sample_compvis_single.py --ckpt PATH_TO_CKPTS --output-path PATH_TO_OUTPUT_IMAGE --prompt "A n image of a gorilla playing the piano in Crayon style."
    ```

* `sampling/stable_diffusion/sample_compvis_automated.py`: this is the automated sampling script used for the machine unlearning experiments. This script will automatically traverse all the styles, object classes, and five different random seeds.
    ```bash
    python3 sampling/stable_diffusion/sample_compvis_automated.py --ckpt PATH_TO_CKPTS --output-path PATH_TO_OUTPUT_IMAGE
    ```
  
#### Sampling Fine-Tuned Image Editing (Instruct-Pix2Pix) Model

To sample from the fine-tuned image editing models, you can use the following scripts.

```bash
python3 sampling/image_editing/sample_compvis_single.py --ckpt PATH_TO_CKPTS --output PATH_TO_OUTPUT_FOLDER --prompt "An image in Crayon style."
```
Note that for image editing, we did not provide automated sampling scripts, as it is not studied in the paper, but you can modify the script above to achieve the same goal.

## Diffusers

We also provide the scripts to train models using diffusers. The following pipelines are mainly from the [official tutorial](https://huggingface.co/docs/diffusers/v0.13.0/en/training/text2image) but with a lot more details:

### Fine-tuning Stable Diffusion Model using Diffuser

1. The default path to the dataset is set to "./data/unlearn_canvas/", which is the same as the path in the compvis pipeline. You can change the path in the scripts if you want to use a different dataset.
3. Train the diffuser. We need to use the `diffuser_sd_training.py` to train the diffuser. An script example is as follows:
  ```bash
  # Note you need to run this script everytime you change the GPU settings, for example if you want to use the different CUDA_VISIBLE_DEVICES settings, you need to re-run this and config the corresponding settings.
  accelerate config
  # You can skip this step if you are already in the ip2p environtment
  pip install git+https://github.com/huggingface/diffusers.git
  
  # Training scripts
  
  export MODEL_NAME="runwayml/stable-diffusion-v1-5"  # We use the same starting model as the one used in the compvis pipeline
  export dataset_name="OPTML-Group/UnlearnCanvas"
  
  accelerate launch diffuser_sd_training.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$dataset_name \
    --use_ema \  # use this
    --resolution=256 --center_crop --random_flip \
    --train_batch_size=32 \  # As large as your server can hold.
    --gradient_accumulation_steps=1 \
#    --gradient_checkpointing \  # please do NOT use this, this can cause memory leak for some reason I have not figured out
    --mixed_precision="fp16" \  # This should be set to the same as the accelerate config
    --max_train_steps=5000 \  # Usually 5000 ~ 10000 is enough 
    --learning_rate=1e-06 \  # Please use 1e-6
    --max_grad_norm=1 \
    --checkpoints_total_limit 5 \  # Save only 5 checkpoints
    --checkpointing_steps 1000 \  # Save checkpoints every 1000 steps
    --lr_scheduler="constant" --lr_warmup_steps=0 \
    --output_dir="logs/diffuser_unlearn_canvas"  # you can change this to your own name
  ```
After training, the relevant checkpoints will be stored in the folder `logs/diffuser_unlearn_canvas`, which corresponds to your `output_dir` parameter. In this folder, you will see folders like `checkpoint-1000`.
You can also find the pretrained checkpoints in the [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`diffusion/diffuser/style50`).

### Sampling with the Fine-Tuned Stable Diffusion Model

 To perform sampling, below is very important:
   * First you need to clone the model repo from [here](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/fp16). You can just use `git clone https://huggingface.co/runwayml/stable-diffusion-v1-5`. After running this, you will have a folder called `stable-diffusion-v1-5`.
   * After cloning, switch to the corresponding branch `fp16` by using `git checkout fp16`, this corresponds to the `--mixed_precision="fp16"` parameter in the training script.
   * Next, we need to use the Git Large File System (git-lfs) and download the large files in this repo. To do this, you need to install git-lfs by using `sudo apt-get install git-lfs` or if you do not have the root access, you can run `conda install git-lfs`.
   * Then, you need to run `git lfs install` and then `git lfs pull` to download the large files.
   * Note that all the steps above only need be done once.

After you finished the procedures above, you should have successfully set up the pipelines needed for sampling. Then, you need to copy everything in the checkpoint folder `logs/diffuser_unlearn_canvas/checkpoint-1000` to the `stable-diffusion-v1-5` folder. This step involves overwriting some files. Then, you can run the following script to perform sampling:
```bash
python3 sampling/stable_diffusion/sample_diffuser_single.py --ckpt PATH_TO_CKPT_FOLDER --output PATH_TO_OUTPUT_IMAGE --prompt "An image of a gorilla playing the piano in Crayon style."
```
or use the following script to perform automated sampling:
```bash
python3 sampling/stable_diffusion/sample_diffuser_automated.py --ckpt PATH_TO_CKPT_FOLDER --output PATH_TO_OUTPUT_DIR
```

## Evaluating Generation Quality

We provide the scripts to evaluate the generated images generated with the script introduced above. The evaluation metrics include the style and object classification accuracy, as well as the FID scores. 

* style classification:
  ```bash
  python3 evaluation/classification.py --task style --input_dir PATH_TO_GENERATED_IMAGES --output_dir PATH_TO_OUTPUT_DIR
  ```
* object classification:
  ```bash
  python3 evaluation/classification.py --task class --input_dir PATH_TO_GENERATED_IMAGES --output_dir PATH_TO_OUTPUT_DIR
  ```
* FID score:
    ```bash
    python3 evaluation/fid.py --p1 PATH_TO_DATASET_FOLDER --p2 PATH_TO_GENERATED_DATASET --output-path PATH_TO_OUTPUT_DIR
    ```