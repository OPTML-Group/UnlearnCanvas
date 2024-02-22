import os
import argparse
import numpy as np
from torchvision import transforms
import torch
torch.hub.set_dir("cache")
from PIL import Image

from tqdm import tqdm

import sys
sys.path.append("")
from evaluation.vgg_loss_utils import prepare_model


theme_available = ["Abstractionism", "Artist_Sketch", "Blossom_Season", "Blue_Blooming", "Bricks", "Byzantine",
                   "Cartoon", "Cold_Warm", "Color_Fantasy", "Comic_Etch", "Crayon", "Crypto_Punks", "Cubism", "Dadaism",
                   "Dapple", "Defoliation", "Dreamweave", "Early_Autumn", "Expressionism", "Fauvism",
                   "Foliage_Patchwork", "French", "Glowing_Sunset", "Gorgeous_Love", "Greenfield", "Impasto",
                   "Impressionism", "Ink_Art", "Joy", "Liquid_Dreams", "Palette_Knife", "Magic_Cube", "Meta_Physics",
                   "Meteor_Shower", "Monet", "Mosaic", "Neon_Lines", "On_Fire", "Pastel", "Pencil_Drawing", "Picasso",
                   "Pointillism", "Pop_Art", "Rainwash", "Realistic_Watercolor", "Red_Blue_Ink", "Rust", "Seed_Images",
                   "Sketch", "Sponge_Dabbed", "Structuralism", "Superstring", "Surrealism", "Techno", "Ukiyoe",
                   "Van_Gogh", "Vibrant_Flow", "Warm_Love", "Warm_Smear", "Watercolor", "Winter"]

class_available = ["Architectures", "Bears", "Birds", "Butterfly", "Cats", "Dogs", "Fishes", "Flame", "Flowers",
                   "Frogs", "Horses", "Human", "Jellyfish", "Rabbits", "Sandwiches", "Sea", "Statues", "Towers",
                   "Trees", "Waterfalls"]


def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--ckpt", type=str, default="ckpts/loss_models/")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create folder if not exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "vgg_loss_align.pth")

    args.vgg = os.path.join(args.ckpt, "vgg_normalised.pth")
    args.decoder_path = os.path.join(args.ckpt, "decoder_iter_160000.pth")
    args.trans_path = os.path.join(args.ckpt, "transformer_iter_160000.pth")
    args.embedding_path = os.path.join(args.ckpt, "embedding_iter_160000.pth")

    model = prepare_model(args.vgg, args.trans_path, args.decoder_path, args.embedding_path, device)

    results = {}
    results["statistics"] = {theme: {obj_class: {"content_loss": [], "style_loss": []} for obj_class in class_available} for
                             theme in theme_available}

    if os.path.exists(output_path):
        results = torch.load(output_path)
        print("Loaded existing results")

    transform = test_transform(512, True)

    # Process each theme
    for idx, test_theme in tqdm(enumerate(theme_available)):
        if test_theme == "Seed_Images":
            continue
        if results["statistics"].get(test_theme, None).get("content_loss_mean", None) is not None:
            print(f"Theme {test_theme} already processed. Skipping...")
            continue
        print(f"Processing theme {test_theme}")
        all_content_losses = []
        all_style_losses = []
        # Process each object class
        for object_class in class_available:
            # batch_result_images = []
            # batch_label_images = []
            # Collect images for the batch
            for test_img_idx in [19, 20]:
                label_image_path = os.path.join(args.benchmark_dir, test_theme, object_class, f"{test_img_idx}.jpg")
                label_image = Image.open(label_image_path)
                label_image_tensor = transform(label_image).to(device).unsqueeze(0)
                for style_img_idx in range(1, 19):
                    img_path = os.path.join(args.input_dir, test_theme,
                                            f"{object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")
                    image = Image.open(img_path)
                    result_image_tensor = transform(image).to(device).unsqueeze(0)
                    with torch.no_grad():
                        content_loss, style_loss = model(result_image_tensor, label_image_tensor)
                        results["statistics"][test_theme][object_class]["content_loss"].append(content_loss.item())
                        results["statistics"][test_theme][object_class]["style_loss"].append(style_loss.item())
                        all_content_losses.append(content_loss.item())
                        all_style_losses.append(style_loss.item())
                    # print(f"Calculated {object_class}_test{test_img_idx}_ref{style_img_idx}.jpg")

            results["statistics"][test_theme][object_class]["content_loss_mean"] = np.mean(results["statistics"][test_theme][object_class]["content_loss"])
            results["statistics"][test_theme][object_class]["style_loss_mean"] = np.mean(results["statistics"][test_theme][object_class]["style_loss"])
            results["statistics"][test_theme][object_class]["content_loss_std"] = np.std(results["statistics"][test_theme][object_class]["content_loss"])
            results["statistics"][test_theme][object_class]["style_loss_std"] = np.std(results["statistics"][test_theme][object_class]["style_loss"])


        results["statistics"][test_theme]["content_loss_mean"] = np.mean(all_content_losses)
        results["statistics"][test_theme]["style_loss_mean"] = np.mean(all_style_losses)
        results["statistics"][test_theme]["content_loss_std"] = np.std(all_content_losses)
        results["statistics"][test_theme]["style_loss_std"] = np.std(all_style_losses)

        # Save the results
        if not args.dry_run:
            torch.save(results, output_path)
