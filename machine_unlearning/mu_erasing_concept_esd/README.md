# Usage
This method is based on the environment `unlearn_canvas`, which is shared across this repository.

## Scripts

The checkpoint used for ESD is in the compvis format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) `compvis/style50/compvis.ckpt`. Some examples are:

```bash
python3 stable_diffusion/train-scripts/train-esd-style.py --prompt "Meteor Shower Style" --train_method xattn --ckpt_path PATH_TO_COMPVIS_CKPT

python3 stable_diffusion/train-scripts/train-esd-class.py --prompt "An image of Dog" --train_method xattn --ckpt_path PATH_TO_COMPVIS_CKPT
```

Note that the parameter `--train_method` should be set to `xattn` for unlearning both styles and objects.