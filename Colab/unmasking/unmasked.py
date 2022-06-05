#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import click
import cv2


@click.group()
def main():
    pass

@main.command("unmask")
#  @click.option("--gpus", type=int, default=0, help="number of GPUs to use")
@click.option("--checkpoint", type=str, default="./models/masks_ver4_model.ckpt")
@click.argument("input_image_path")
@click.argument("mask_image_path")
def unmask(input_image_path, mask_image_path, checkpoint):
    import random
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from PIL import Image
    from network import SNPatchGAN
    from dataset import MaskedCelebADataset

    model_checkpoint_path = Path(checkpoint)
    if not model_checkpoint_path.exists():
        print("checkpoint path doesn't exist")
        os.exit(1)

    #  # load images
    #  input_img = Image.open(input_image_path)
    #  mask_img = Image.open(mask_image_path)
    #  # resize+crop
    #  input_img = transforms.Resize((128,128))(input_img)
    #  input_img = transforms.CenterCrop((128,128))(input_img)
    #  mask_img = transforms.Resize((128,128))(mask_img)
    #  mask_img = transforms.CenterCrop((128,128))(mask_img)
    #  # to tensor
    #  input_img = transforms.ToTensor()(input_img)
    #  mask_img = transforms.ToTensor()(mask_img)[0].unsqueeze(dim=0)
    #  # normalize img
    #  #  input_img = transforms.Normalize(mean, std)
    #  # 4D
    #  input_img = input_img.unsqueeze(dim=0)
    #  mask_img = mask_img.unsqueeze(dim=0)

    # load model
    model = SNPatchGAN()
    model.load_from_checkpoint(checkpoint)
    model.eval()
    model.freeze()

    # get img
    ds = MaskedCelebADataset("dataset/celeba", (256, 256), mode='eval', apply_transforms=True)
    dl = DataLoader(ds, batch_size=6, drop_last=True, shuffle=False)
    img, masked_img, mask = None, None, None
    for batch in dl:
        img, masked_img, mask = batch
        break
    # inference
    coarse, refined = model(masked_img, mask)
    completed = img * (1 - mask) + refined * mask

    # show
    refined_img = transforms.ToPILImage()(refined[0])
    refined_img.show()

@main.command("export-model")
@click.argument("checkpoint", type=str, default="./models/masks_ver3_model.ckpt")
@click.argument("output", default=Path(os.getcwd()) / "unmasked-model.onnx")
def export_model(checkpoint, output):
    import torch
    import torch.nn as nn
    from network import SNPatchGAN

    model_checkpoint_path = Path(checkpoint)
    if not model_checkpoint_path.exists():
        print("checkpoint path doesn't exist")
        sys.exit(1)

    def remove_all_spectral_norm(item):
        if isinstance(item, nn.Module):
            try:
                nn.utils.remove_spectral_norm(item)
            except Exception:
                pass

            for child in item.children():
                remove_all_spectral_norm(child)

        if isinstance(item, nn.ModuleList):
            for module in item:
                remove_all_spectral_norm(module)

        if isinstance(item, nn.Sequential):
            modules = item.children()
            for module in modules:
                remove_all_spectral_norm(module)

    model = SNPatchGAN()
    model.load_from_checkpoint(checkpoint)
    #  torch.nn.utils.remove_spectral_norm(model)
    remove_all_spectral_norm(model)
    model.freeze()
    model.eval()

    model.to_onnx(
        output,
        input_sample=(torch.ones((1, 3, 128, 128)), torch.ones((1, 1, 128, 128))), # output: coarse, refined
        example_outputs=(torch.ones((1, 3, 128, 128)), torch.ones((1, 3, 128, 128))), # output: coarse, refined
        export_params=True,
        opset_version=13,
    )
    #  script = model.to_torchscript()
    #  torch.jit.save(script, output)


@main.command("mask-face")
@click.option("--mask-type", type=str, default="random", show_default=True, help="Type of face mask to apply to the image")
@click.argument("image")
@click.argument("output_dir", default=os.getcwd())
def mask_face(image, mask_type, output_dir):
    import mask_the_face

    output_dir_path = Path(output_dir)
    image_path = Path(image)
    if not output_dir_path.exists():
        print("output directory doesn't exist")
        os.exit(1)
    if not image_path.exists():
        print("image not found")
        os.exit(1)

    masker = mask_the_face.Masker()
    try:
        masked_image, mask, _, _ = masker.apply_mask_file(str(image_path), mask_type=mask_type)
    except Exception as e:
        print("face masking failed")
        os.exit(2)

    masked_image_filename = image_path.stem + "-face_masked" + image_path.suffix
    mask_image_filename = image_path.stem + "-mask" + image_path.suffix

    cv2.imwrite(str(output_dir_path / masked_image_filename), masked_image)
    cv2.imwrite(str(output_dir_path / mask_image_filename), mask)

#  main.add_command(mask_face)

if __name__ == "__main__":
    main()
