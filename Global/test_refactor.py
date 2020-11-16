# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from .options.test_options import TestOptions
from .models.models import create_model
from .models.mapping_model import Pix2PixHDModel_Mapping
from pathlib import Path
import Global.util.util as util
from PIL import Image
import torch
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import numpy as np


def _data_transforms(img, method=Image.BILINEAR, scale=False):
    ow, oh = img.size
    pw, ph = ow, oh
    if scale == True:
        if ow < oh:
            ow = 256
            oh = ph / pw * 256
        else:
            oh = 256
            ow = pw / ph * 256

    h = int(round(oh / 4) * 4)
    w = int(round(ow / 4) * 4)

    if (h == ph) and (w == pw):
        return img

    return img.resize((w, h), method)


def _data_transforms_rgb_old(img):
    w, h = img.size
    A = img
    if w < 256 or h < 256:
        A = transforms.Scale(256, Image.BILINEAR)(img)
    return transforms.CenterCrop(256)(A)


def _irregular_hole_synthesize(img, mask):
    img_np = np.array(img).astype("uint8")
    mask_np = np.array(mask).astype("uint8")
    mask_np = mask_np / 255
    img_new = img_np * (1 - mask_np) + mask_np * 255

    hole_img = Image.fromarray(img_new.astype("uint8")).convert("RGB")

    return hole_img


def _parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.label_nc = 0
    opt.n_downsample_global = 3
    opt.mc = 64
    opt.k_size = 4
    opt.start_r = 1
    opt.mapping_n_block = 6
    opt.map_mc = 512
    opt.no_instance = True
    opt.checkpoints_dir = "./checkpoints/restoration"
    ##

    if opt.Quality_restore:
        _quality_config(opt)
    if opt.Scratch_and_Quality_restore:
        _scratch_and_quality_config(opt)


def _scratch_and_quality_config(opt):
    opt.NL_res = True
    opt.use_SN = True
    opt.correlation_renormalize = True
    opt.NL_use_mask = True
    opt.NL_fusion_method = "combine"
    opt.non_local = "Setting_42"
    opt.name = "mapping_scratch"
    opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
    opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_scratch")


def _quality_config(opt):
    opt.name = "mapping_quality"
    opt.load_pretrainA = os.path.join(opt.checkpoints_dir, "VAE_A_quality")
    opt.load_pretrainB = os.path.join(opt.checkpoints_dir, "VAE_B_quality")


def repair(input_dir: Path, output_dir: Path, scratched=False):
    opt = TestOptions().parse(save=False)
    _parameter_set(opt)

    model = Pix2PixHDModel_Mapping()

    model.initialize(opt)  #TODO refactor to take dictionary?
    model.eval()

    input_loader: [Path] = [path for path in input_dir.iterdir()]
    input_loader.sort()  # sorted purely for printing purposes

    if scratched:
        mask_loader = [path for path in (output_dir / 'stage_1_restore_output/masks/mask').iterdir()]
        mask_loader.sort()
        dataset_size = len(mask_loader)
    else:
        dataset_size = len(input_loader)

    img_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    mask_transform = transforms.ToTensor()

    for i in range(dataset_size):

        input_img = Image.open(input_loader[i]).convert("RGB")

        print("Now you are processing %s".format(input_loader[i].name))

        if scratched:
            mask = Image.open(mask_loader[i]).convert("RGB")
            origin = input_img
            input_img = _irregular_hole_synthesize(input_img, mask)
            mask = mask_transform(mask)[:1, :, :]
            mask = mask[:1, :, :]  ## Convert to single channel
            mask = mask.unsqueeze(0)
            input_img = img_transform(input_img)
            input_img = input_img.unsqueeze(0)
        else:
            input_img = _data_transforms(input_img, scale=False)  # consider full res version only
            origin = input_img
            input_img = img_transform(input_img)
            input_img = input_img.unsqueeze(0)
            mask = torch.zeros_like(input_img)
        ### Necessary input

        try:
            generated = model.inference(input_img, mask)
        except Exception as e:
            print("Skip %s".format(input_loader[i]))
            print(e)
            continue

        if str(input_loader[i]).endswith(".jpg"):
            input_loader[i] = input_loader[i][:-4] + ".png"

        image_grid = vutils.save_image(
            (input_img + 1.0) / 2.0,
            output_dir / "input_image" / input_loader[i].name,
            nrow=1,
            padding=0,
            normalize=True,
        )
        image_grid = vutils.save_image(
            (generated.data.cpu() + 1.0) / 2.0,
            output_dir / "restored_image" / input_loader[i].name,
            nrow=1,
            padding=0,
            normalize=True,
        )

        origin.save(output_dir / "origin" / input_loader[i].name)
