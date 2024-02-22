# Implementation Notes for ArtFlow

## Resources

Code: https://github.com/diyiiyiii/StyTR-2
Paper: https://github.com/diyiiyiii/StyTR-2#stytr2--image-style-transfer-with-transformerscvpr2022 (CPVR 2022)

## Preparation

Download available ckpts:
vgg-model: https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view?usp=sharing, 
vit_embedding: https://drive.google.com/file/d/1C3xzTOWx8dUXXybxZwmjijZN8SrC3e4B/view?usp=sharing, 
decoder: https://drive.google.com/file/d/1fIIVMTA_tPuaAAFtqizr6sd1XV7CX6F9/view?usp=sharing, 
Transformer_module: https://drive.google.com/file/d/1dnobsaLeE889T_LncCkAA2RkqzwsfHYy/view?usp=sharing.

## Environment

This code is running in the environment `unlearn_canvas`.


## Code Running

Automated batch test.
```bash
python3 unlearn_canvas_batch_test.py --output_dir ./eval_results/style_transfer/stytr2/style60 --img_dir PATH_TO_DATASET_DIR
```