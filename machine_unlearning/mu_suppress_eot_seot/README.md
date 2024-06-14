# Usage

The experiments in this folder are carried in the environment `seot`. 
```bash
conda create -n seot python=3.8
conda activate seot
pip install -r requirements.txt
```


## Unlearning and Generating

- SEOT is a method which uses image editing to **unlearn** styles/objects in generated images.
- Simply use the original model to generate the unlearned image.
- The checkpoint used for SEOT is in the diffuser format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`diffuser/style50`).

```bash
cd ../evaluation

# for object
python sampling_unlearned_models/seot.py --theme ${theme} --type object --ckpt_path PATH_TO_COMPVIS_CKPT

# for style
python sampling_unlearned_models/seot.py --theme ${theme} --type style --ckpt_path PATH_TO_COMPVIS_CKPT
```