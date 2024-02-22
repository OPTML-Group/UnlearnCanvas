# Usage

This repo is based on the environment `unlearn_canvas`.

The checkpoint used for FMN is in the diffuser format, and the pretrained checkpoint can be found in [Google Drive](https://drive.google.com/drive/folders/14iztBXs-GoBFVLePC2_psP00YUMK5-cy?usp=sharing) (`diffuser/style50`). 

This method is super-fast but the coefficients are difficult to tune. The following arguments are suggested to run the experiments:

```
for theme in Abstractionism Artist_Sketch Blossom_Season Bricks Byzantine Cartoon Cold_Warm Color_Fantasy Comic_Etch Crayon Cubism Dadaism Dapple Defoliation Early_Autumn Expressionism Fauvism French Glowing_Sunset Gorgeous_Love Greenfield Impressionism Ink_Art Joy Liquid_Dreams Magic_Cube Meta_Physics Meteor_Shower Monet Mosaic Neon_Lines On_Fire Pastel Pencil_Drawing Picasso Pop_Art Red_Blue_Ink Rust Seed_Images Sketch Sponge_Dabbed Structuralism Superstring Surrealism Ukiyoe Van_Gogh Vibrant_Flow Warm_Love Warm_Smear Watercolor Winter; do python3 train_erase.py --ckpt PATH_TO_DIFFUSER_DIR --theme ${theme} --output_dir OUTPUT_DIR  --erase_scale 0.05 --lamb 1.0 --guided_concepts "An image in Photo style" --add_prompts; done


for topic in Architectures Bears Birds Butterfly Cats Dogs Fishes Flame Flowers Frogs Horses Human Jellyfish Rabbits Sandwiches Sea Statues Towers Trees Waterfalls; do python3 train_erase.py --ckpt PATH_TO_DIFFUSER_DIR --theme ${theme} --output_dir results/style50/  --erase_scale 0.01 --lamb 10.0 --guided_concepts "A Elephant image" --add_prompts; done
```