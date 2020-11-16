# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import shutil
from pathlib import Path

from Global.test_refactor import repair


def print_run_stage(n: int, stage_name: str) -> None:
    print("Running Stage {}: {}".format(n, stage_name))


def print_stage(n: int) -> None:
    print("Finish stage {}...\n".format(n))


def mk_output_dirs(output_dir: Path) -> (Path, Path, Path, Path):
    stage_1_out_dir = output_dir / "stage_1_restore_output"
    stage_2_out_dir = output_dir / "stage_2_detection_output"
    stage_3_out_dir = output_dir / "stage_3_face_output"
    final_output = output_dir / "final_output"

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

    repair(input_folder, output_folder, scratched=bool(opts.with_scratch))
    # TODO determine if model can be freed after each phase to reduce VRAM requirements
    # Solve the case when there is no face in the old photo
    for image in (stage_1_output_dir / "restored_image").iterdir():
        shutil.copy(image, output_folder / "final_output")

    print_stage(1)

    # Stage 2: Face Detection
    print_run_stage(2, "Face Detection")
    # os.chdir(".././Face_Detection")
    # stage_2_input_dir = stage_1_output_dir / "restored_image"
    # stage_2_output_dir = Path(opts.output_folder) / "stage_2_detection_output"
    # Path.mkdir(stage_2_output_dir, parents=True, exist_ok=True)

    # run detect_all_dlib.py --url " + stage_2_input_dir + " --save_url " + stage_2_output_dir

    print_stage(2)

    # Stage 3: Face Restore
    print_run_stage(3, 'Face Enhancement')
    # os.chdir(".././Face_Enhancement")
    # stage_3_input_mask = "./"
    # stage_3_input_face = stage_2_output_dir
    # stage_3_output_dir = os.path.join(opts.output_folder, "stage_3_face_output")

    # run test_face.py

    print_stage(3)

    # Stage 4: Warp back
    print_run_stage(4, 'Blending')
    os.chdir(".././Face_Detection")

    # run  align_warp_back_multiple_dlib.py
    print_stage(4)

    print("All the processing is done. Please check the results.")
