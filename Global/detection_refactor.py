# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import argparse
import time

import torch
import torchvision as tv
import torch.nn.functional as F
from detection_util.util import *
from detection_models import networks
from PIL import Image, ImageFile
from pathlib import Path
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


def data_transforms(img, full_size, method=Image.BICUBIC):
    if full_size == "full_size":
        ow, oh = img.size
        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == oh) and (w == ow):
            return img
        return img.resize((w, h), method)

    if full_size == "resize_256":
        return img.resize((config.image_size, config.image_size), method)

    if full_size == "scale_256":

        ow, oh = img.size
        pw, ph = ow, oh
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

        h = int(round(oh / 16) * 16)
        w = int(round(ow / 16) * 16)
        if (h == ph) and (w == pw):
            return img
        return img.resize((w, h), method)


def blend_mask(img, mask):

    np_img = np.array(img).astype("float")

    return Image.fromarray((np_img * (1 - mask) + mask * 255.0).astype("uint8")).convert("RGB")


def detect_scratches(gpu: int, test_path: Path, output_dir: Path, input_dir: Path):
    print("initializing the dataloader")

    model = networks.UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        sync_bn=True,
        antialiasing=True,
    )

    ## load model
    checkpoint_path = "./checkpoints/detection/FT_Epoch_latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    print("model weights loaded")

    model.to(gpu)
    model.eval()

    ## dataloader and transformation
    print("directory of testing image: " + test_path)
    imagelist = [path for path in test_path.iterdir()]
    imagelist.sort()

    input_dir = output_dir / 'input'
    output_dir = output_dir / 'mask'
    idx = 0

    for im_path in imagelist:

        idx += 1

        print("processing", im_path)

        scratch_image = Image.open(test_path / im_path.name).convert("RGB")
        scratch_image = Image.open(test_path / im_path.name)

        transformed_image_PIL = data_transforms(scratch_image, 'full_size')

        scratch_image = transformed_image_PIL.convert("L")
        scratch_image = tv.transforms.ToTensor()(scratch_image)

        scratch_image = tv.transforms.Normalize([0.5], [0.5])(scratch_image)

        scratch_image = torch.unsqueeze(scratch_image, 0).to(gpu)

        P = torch.sigmoid(model(scratch_image))

        P = P.data.cpu()

        tv.utils.save_image(
            (P >= 0.4).float(),
            os.path.join(output_dir, im_path[:-4] + ".png",),
            nrow=1,
            padding=0,
            normalize=True,
        )
        transformed_image_PIL.save(os.path.join(input_dir, im_path[:-4] + ".png"))
