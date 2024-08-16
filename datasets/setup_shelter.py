#!/usr/bin/env python3

import os
import sys
import numpy as np
import shutil
import colmap_loader as cldr
from colmap_loader import qvec2rotmat
import dataset_util as dutil
from types import SimpleNamespace


scene_url = "https://sagemaker-studio-163033260074-0o5wme941ix9.s3.ap-southeast-2.amazonaws.com/shelter.zip"
scene_file = "shelter.zip"
scene_name = "shelter"
ACE_DIRS = SimpleNamespace(
    train=SimpleNamespace(
        rgb="train/rgb/",
        calib="train/calibration/",
        pose="train/poses/",
    ),
    test=SimpleNamespace(
        rgb="test/rgb/",
        calib="test/calibration/",
        pose="test/poses/",
    ),
)


def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)


def train80test20split(idx, image_name, total):
    if idx < total * 0.8:
        return ACE_DIRS.train
    else:
        return ACE_DIRS.test


def download_unzip_dataset():
    print("Downloading and unzipping data...")
    os.system("wget " + scene_url)
    os.system(f"unzip -q {scene_file} -d {scene_name}")
    os.system("rm " + scene_file)


def read_colmap_binary_data():
    print("Loading COLMAP binary data...")
    images_bin = "sparse/0/images.bin"
    camera_bin = "sparse/0/cameras.bin"
    camera_poses = cldr.read_extrinsics_binary(images_bin)
    camera_calibs = cldr.read_intrinsics_binary(camera_bin)

    print(
        f"Reconstruction contains {len(camera_poses)} cameras \
 and {len(camera_calibs)} camera calibration(s)."
    )
    return camera_poses, camera_calibs


def create_ace_dirs():
    print("Creating ACE dir structure...")
    dutil.mkdir(ACE_DIRS.train.rgb)
    dutil.mkdir(ACE_DIRS.train.calib)
    dutil.mkdir(ACE_DIRS.train.pose)
    dutil.mkdir(ACE_DIRS.test.rgb)
    dutil.mkdir(ACE_DIRS.test.calib)
    dutil.mkdir(ACE_DIRS.test.pose)
    return ACE_DIRS


def convert_colmap_data(cam_extrinsics, cam_intrinsics, images_folder, mode_dir_fn):
    for idx, key in enumerate(cam_extrinsics):
        print()
        # the exact output you're looking for:
        print("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_file = os.path.basename(extr.name)
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        cam_pose = getWorld2View(R, T)
        inv_cam_pose = np.linalg.inv(cam_pose)

        mode_dir = mode_dir_fn(idx, image_file, len(cam_extrinsics))
        # Copy file to rgb dir
        print(f"Copy {image_path} to {mode_dir.rgb} dir...")
        shutil.copy2(image_path, mode_dir.rgb)

        # Write camera pose to pose dir
        print(f"Write camera pose to {mode_dir.pose} dir...")
        with open(mode_dir.pose + image_file[:-3] + "txt", "w") as f:
            f.write(
                str(float(inv_cam_pose[0, 0]))
                + " "
                + str(float(inv_cam_pose[0, 1]))
                + " "
                + str(float(inv_cam_pose[0, 2]))
                + " "
                + str(float(inv_cam_pose[0, 3]))
                + "\n"
            )
            f.write(
                str(float(inv_cam_pose[1, 0]))
                + " "
                + str(float(inv_cam_pose[1, 1]))
                + " "
                + str(float(inv_cam_pose[1, 2]))
                + " "
                + str(float(inv_cam_pose[1, 3]))
                + "\n"
            )
            f.write(
                str(float(inv_cam_pose[2, 0]))
                + " "
                + str(float(inv_cam_pose[2, 1]))
                + " "
                + str(float(inv_cam_pose[2, 2]))
                + " "
                + str(float(inv_cam_pose[2, 3]))
                + "\n"
            )
            f.write(
                str(float(inv_cam_pose[3, 0]))
                + " "
                + str(float(inv_cam_pose[3, 1]))
                + " "
                + str(float(inv_cam_pose[3, 2]))
                + " "
                + str(float(inv_cam_pose[3, 3]))
                + "\n"
            )

        # Write calibration to calib dir
        print(f"Write camera focal length to {mode_dir.calib} dir...")
        with open(mode_dir.calib + image_file[:-3] + "txt", "w") as f:
            f.write(str(focal_length_x))


if __name__ == "__main__":

    print("===== Processing " + scene_name + " ===================")
    download_unzip_dataset()
    os.chdir(scene_name)
    extrinsics, intrinsics = read_colmap_binary_data()

    frames_dir = "frames"
    create_ace_dirs()

    convert_colmap_data(extrinsics, intrinsics, frames_dir, train80test20split)
