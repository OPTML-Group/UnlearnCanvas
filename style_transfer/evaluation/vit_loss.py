import os
import argparse
import numpy as np
import timm
from torchvision import transforms
import torch
torch.hub.set_dir("cache")
from PIL import Image

from tqdm import tqdm
import torch.nn.functional as F


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


@torch.no_grad()
def calc_style_loss(vit_model, image1_tensor, image2_tensor):
    # Function to extract the required feature maps
    def get_feature_maps(model, image, layers):
        handles = []
        feature_maps = []

        # Register a forward hook to capture feature maps
        def hook_fn(module, input, output):
            feature_maps.append(output)

        # Attach hooks to the specified layers
        for layer in layers:
            handle = model.blocks[layer].register_forward_hook(hook_fn)
            handles.append(handle)

        # Forward pass
        with torch.no_grad():
            model(image)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return feature_maps

    # Specify transformer blocks (0-indexed, hence 1, 3, 5, etc.)
    transformer_blocks = range(3, 24, 4)

    # Get feature maps for both images
    features_image1 = get_feature_maps(vit_model, image1_tensor, transformer_blocks)
    features_image2 = get_feature_maps(vit_model, image2_tensor, transformer_blocks)

    # Calculate and average the mean square error
    style_loss = 0
    for f1, f2 in zip(features_image1, features_image2):
        style_loss += F.mse_loss(f1, f2)

    return style_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--benchmark_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create folder if not exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "vit_loss_align.pth")

    model = timm.create_model("vit_large_patch16_224.augreg_in21k").to(device)
    model.head = torch.nn.Linear(1024, len(theme_available)).to(device)

    # load checkpoint
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
    model.eval()

    results = {}
    results["statistics"] = {theme: {obj_class: {"content_loss": [], "style_loss": []} for obj_class in class_available} for
                             theme in theme_available}

    if os.path.exists(output_path):
        results = torch.load(output_path)
        print("Loaded existing results")

    transform = test_transform(224, True)

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
                        # content_loss, style_loss = model(result_image_tensor, label_image_tensor)
                        style_loss = calc_style_loss(model, result_image_tensor, label_image_tensor)
                        results["statistics"][test_theme][object_class]["style_loss"].append(style_loss.item())
                        all_style_losses.append(style_loss.item())

        # Save the results
        if not args.dry_run:
            torch.save(results, output_path)
