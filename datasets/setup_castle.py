#!/usr/bin/env python3

import os
from dataclasses import dataclass
import numpy as np
import shutil
import dataset_util as dutil
from types import SimpleNamespace
from scipy.spatial.transform import Rotation


scene_url = "https://sagemaker-studio-163033260074-0o5wme941ix9.s3.ap-southeast-2.amazonaws.com/castle.zip"
scene_file = "castle.zip"
scene_name = "castle"

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

UE_CAMERA_CALIB = SimpleNamespace(
    fov=60.0, height=1000, width=1500
)

@dataclass
class UECameraPose:
    image_name: str
    x: float
    y: float
    z: float
    pitch: float
    roll: float
    yaw: float



def getView2World(x,y,z, pitch, roll, yaw):
    R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = [x, y, z]
    Rt[3, 3] = 1.0
    return np.float32(Rt)



def train_test_split_random(idx, image_name, total):
    if np.random.rand() < 0.2:
        return ACE_DIRS.test
    else:
        return ACE_DIRS.train


SCENE_SPLIT_FNS = [
    train_test_split_random,
]
SCENE_SPLIT_TYPES = ["i20r"]


def download_unzip_dataset():
    print("Downloading and unzipping data...")
    os.system("wget " + scene_url)
    os.system(f"unzip -q {scene_file} -d {scene_name}")
    os.system("rm " + scene_file)


def read_ue_capture_data():
    print("Loading UE caputured images data...")

    # calculate focal length from fov in degree and sensor width
    focal = UE_CAMERA_CALIB.width / (2 * np.tan(UE_CAMERA_CALIB.fov * np.pi / 360))

    camera_file = "paths/semi_sphere_paths.txt"
    # read each line from cameara file. 
    # The line is like "image_000001.jpeg, x, y, z, pitch, roll, yaw". 
    # Split each line by comma and convert to UECameraPose objects
    camera_poses = []
    camera_focals = []
    
    with open(camera_file, 'r') as file:
        for line in file:
            line_parts = line.strip().split(',')
            image_name = line_parts[0]
            x, y, z, pitch, roll, yaw = map(float, line_parts[1:])
            camera_pose = UECameraPose(image_name, x, y, z, pitch, roll, yaw)
            camera_poses.append(camera_pose)

            camera_focals.append(focal)


    print(
        f"Reconstruction contains {len(camera_poses)} cameras \
 and {len(camera_focals)} camera focals."
    )
    return camera_poses, camera_focals


def create_ace_dirs(scene_dir):
    print("Creating ACE dir structure...")
    dutil.mkdir(scene_dir + ACE_DIRS.train.rgb)
    dutil.mkdir(scene_dir + ACE_DIRS.train.calib)
    dutil.mkdir(scene_dir + ACE_DIRS.train.pose)
    dutil.mkdir(scene_dir + ACE_DIRS.test.rgb)
    dutil.mkdir(scene_dir + ACE_DIRS.test.calib)
    dutil.mkdir(scene_dir + ACE_DIRS.test.pose)


def convert_ue_data(
    cam_extrinsics, camera_focals, images_folder, scene_dir, mode_dir_fn
):
    for idx, extr in enumerate(cam_extrinsics):
        print()
        # the exact output you're looking for:
        print("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))

        focal_length = camera_focals[idx]

        image_file = extr.image_name
        image_path = os.path.join(images_folder, image_file)
        cam_pose = getView2World(extr.x, extr.y, extr.z, extr.pitch, extr.roll, extr.yaw)

        mode_dir = mode_dir_fn(idx, image_file, len(cam_extrinsics))
        # Copy file to rgb dir
        print(f"Copy {image_path} to {scene_dir + mode_dir.rgb} dir...")
        shutil.copy2(image_path, scene_dir + mode_dir.rgb)

        # Write camera pose to pose dir
        print(f"Write camera pose to {scene_dir + mode_dir.pose} dir...")
        with open(scene_dir + mode_dir.pose + image_file[:-4] + "txt", "w") as f:
            f.write(
                str(float(cam_pose[0, 0]))
                + " "
                + str(float(cam_pose[0, 1]))
                + " "
                + str(float(cam_pose[0, 2]))
                + " "
                + str(float(cam_pose[0, 3]))
                + "\n"
            )
            f.write(
                str(float(cam_pose[1, 0]))
                + " "
                + str(float(cam_pose[1, 1]))
                + " "
                + str(float(cam_pose[1, 2]))
                + " "
                + str(float(cam_pose[1, 3]))
                + "\n"
            )
            f.write(
                str(float(cam_pose[2, 0]))
                + " "
                + str(float(cam_pose[2, 1]))
                + " "
                + str(float(cam_pose[2, 2]))
                + " "
                + str(float(cam_pose[2, 3]))
                + "\n"
            )
            f.write(
                str(float(cam_pose[3, 0]))
                + " "
                + str(float(cam_pose[3, 1]))
                + " "
                + str(float(cam_pose[3, 2]))
                + " "
                + str(float(cam_pose[3, 3]))
                + "\n"
            )

        # Write calibration to calib dir
        print(f"Write camera focal length to {scene_dir + mode_dir.calib} dir...")
        with open(scene_dir + mode_dir.calib + image_file[:-4] + "txt", "w") as f:
            f.write(str(focal_length))


if __name__ == "__main__":
    
    split_types = SCENE_SPLIT_TYPES

    print("===== Processing " + scene_name + " ===================")
    download_unzip_dataset()
    os.chdir(scene_name)
    extrinsics, intrinsics = read_ue_capture_data()

    os.chdir("..")
    frames_dir = f"{scene_name}/images"
    for type in split_types:
        print("===== Split type: " + type + " ===================")
        scene_dir = f"{scene_name}_{type}/"
        create_ace_dirs(scene_dir)

        split_fn = SCENE_SPLIT_FNS[SCENE_SPLIT_TYPES.index(type)]
        convert_ue_data(extrinsics, intrinsics, frames_dir, scene_dir, split_fn)

    os.system("rm -rf " + scene_name)
