# Usage

The experiments in this folder are carried in the environment `spm`. 
```bash
conda create -n spm python=3.10
conda activate spm
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xformers
pip install -r requirements.txt
```


## Unlearning

The checkpoint used for SPM is in the diffuser format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`diffuser/style50`).

```bash
python train-erase.py --theme ${theme} --ckpt_path PATH_TO_COMPVIS_CKPT
```

The results will be saved in the directory `output/{theme}/`.