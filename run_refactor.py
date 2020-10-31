# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import shutil
from pathlib import Path


def print_run_stage(n: int, stage_name: str):
    print("Running Stage {}: {}".format(n, stage_name))


def print_stage(n: int) -> None:
    print("Finish stage {}...\n".format(n))


def mk_output_dirs(output_folder: Path) -> (Path, Path, Path, Path):
    stage_1_out_dir = output_folder / "stage_1_restore_output"
    stage_2_out_dir = output_folder / "stage_2_detection_output"
    stage_3_out_dir = output_folder / "stage_3_face_output"
    final_output = output_folder / "final_output"

    (stage_3_out_dir / 'each_img').mkdir(parents=True, exist_ok=True)
    for p in (stage_1_out_dir, stage_2_out_dir, final_output):
        p.mkdir(parents=True, exist_ok=True)

    for p in ('input_image', 'masks/input', 'masks/mask', 'origin', 'restored_image'):
        (stage_1_out_dir / p).mkdir(parents=True, exist_ok=True)

    return stage_1_out_dir, stage_2_out_dir, stage_3_out_dir, final_output


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="imgs/input_folder", help="Images to restore")
    parser.add_argument(
        "--output_folder",
        type=str,
        default="imgs/output_folder",
        help="Restored images",
    )
    parser.add_argument("--GPU", type=str, default="6,7", help="0,1,2")
    parser.add_argument(
        "--checkpoint_name", type=str, default="Setting_9_epoch_100", help="choose which checkpoint"
    )
    parser.add_argument("--with_scratch", action="store_true")
    opts = parser.parse_args()

    gpu1 = opts.GPU

    output_folder = Path(opts.output_folder)
    input_folder = Path(opts.input_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    stage_1_output_dir, \
    stage_2_output_dir, \
    stage_3_output_dir, \
    final_output_dir = mk_output_dirs(output_folder)

    # Stage 1: Overall Quality Improve
    print_run_stage(1, 'Overall restoration')
    stage_1_input_dir = Path(opts.input_folder)
    stage_1_output_dir = Path(opts.output_folder) / "stage_1_restore_output"
    Path.mkdir(stage_1_output_dir, parents=True, exist_ok=True)

    if not opts.with_scratch:
    # run test.py functions without scratch repair
    else:
    # run test.py with scratch repair

    # Solve the case when there is no face in the old photo
    stage_1_results = Path(stage_1_output_dir) / "restored_image"
    stage_4_output_dir = Path(opts.output_folder) / "final_output"
    Path.mkdir(stage_4_output_dir, parents=True, exist_ok=True)
    for image in stage_1_results.iterdir():
        shutil.copy(image, stage_4_output_dir)

    print_stage(1)

    # Stage 2: Face Detection
    print_run_stage(2, "Face Detection")
    os.chdir(".././Face_Detection")
    stage_2_input_dir = stage_1_output_dir / "restored_image"
    stage_2_output_dir = Path(opts.output_folder) / "stage_2_detection_output"
    Path.mkdir(stage_2_output_dir, parents=True, exist_ok=True)

    stage_2_command = (
            "python3 detect_all_dlib.py --url " + stage_2_input_dir + " --save_url " + stage_2_output_dir
    )
    os.system(stage_2_command)
    print_stage(2)

    # Stage 3: Face Restore
    print_run_stage(3, 'Face Enhancement')
    os.chdir(".././Face_Enhancement")
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)
    # run test_face.py

    print_stage(3)

    # Stage 4: Warp back
    print_run_stage(4, 'Blending')
    os.chdir(".././Face_Detection")
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(opts.output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    stage_4_command = (
            "python3 align_warp_back_multiple_dlib.py --origin_url "
            + stage_4_input_image_dir
            + " --replace_url "
            + stage_4_input_face_dir
            + " --save_url "
            + stage_4_output_dir
    )
    os.system(stage_4_command)
    print_stage(4)

    print("All the processing is done. Please check the results.")
