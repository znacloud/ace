#!/usr/bin/env python3

import os
from dataclasses import dataclass
import numpy as np
import shutil
import dataset_util as dutil
from types import SimpleNamespace
from scipy.spatial.transform import Rotation


scene_url = "https://sagemaker-studio-163033260074-0o5wme941ix9.s3.ap-southeast-2.amazonaws.com/castle_p30.zip"
scene_file = "castle_p30.zip"
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
    fov=60.0, height=720, width=1080
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
    R = Rotation.from_euler('YXZ', [pitch, roll, yaw], degrees=True).as_matrix()

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = [x, y, z]
    Rt[3, 3] = 1.0
    return np.float32(Rt)


# seq1 train: r4000
# seq2 train: r6000
# seq3 train: r4000, r4500
# seq4 train: r5500, r6000
# seq5 train: r4000, r4500, r5500, r6000
# seq6 train: random50%
# seq7 train: random100%
# seq8 train: random100% + seq5
def train_test_split_1(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif image_name.startswith("image_n360_r4000") :
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_2(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif image_name.startswith("image_n360_r6000") :
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_3(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif image_name.startswith("image_n360_r4000") or \
        image_name.startswith("image_n360_r4500"):
        return ACE_DIRS.train
    else:
        return False

def train_test_split_4(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif image_name.startswith("image_n360_r5500") or \
        image_name.startswith("image_n360_r6000"):
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_5(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif image_name.startswith("image_n360_"):
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_6(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif not image_name.startswith("image_n360_") and np.random.rand() > 0.5:
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_7(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    elif not image_name.startswith("image_n360_"):
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_8(idx, image_name, total):
    if image_name.startswith("image_n360_r5000"):
        return ACE_DIRS.test
    else:
        return ACE_DIRS.train



SCENE_SPLIT_FNS = [
    train_test_split_1,
    train_test_split_2,
    train_test_split_3,
    train_test_split_4,
    train_test_split_5,
    train_test_split_6,
    train_test_split_7,
    train_test_split_8,
]
SCENE_SPLIT_TYPES = ["seq1","seq2","seq3","seq4","seq5","seq6","seq7", "seq8"]


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
    cam_extrinsics, camera_focals, image_paths, scene_dir, mode_dir_fn
):
    for idx, extr in enumerate(cam_extrinsics):
        print()
        # the exact output you're looking for:
        print("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        
        image_file = extr.image_name
        mode_dir = mode_dir_fn(idx, image_file, len(cam_extrinsics))
        if not mode_dir:
            print(f"Skipping {image_file}")
            continue

        focal_length = camera_focals[idx]

        
        image_path = os.path.join(image_paths[idx], image_file)
        # UE Coordinate System: left-handed (Z-Up, X-Forward, Y-Right)
        # Convert it to COLMAP coordinate system: right-handed (Z-Forward, X-Right, Y-Down)
        x = extr.y
        y = -extr.z
        z = extr.x
        pitch = extr.yaw
        roll = -extr.pitch
        yaw = extr.roll

        cam_pose = getView2World(x, y, z, pitch, roll, yaw)

        
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
    np.random.seed(42)
    split_types = SCENE_SPLIT_TYPES

    print("===== Processing " + scene_name + " ===================")
    download_unzip_dataset()
    os.chdir("castle_p30")
    image_paths = []
    extrinsics, intrinsics = read_ue_capture_data()
    image_paths.extend(["castle_p30/images" for _ in range(len(extrinsics))])
    os.chdir("../castle")
    extmp, intmp = read_ue_capture_data()
    extrinsics.extend(extmp)
    intrinsics.extend(intmp)
    image_paths.extend(["castle/images" for _ in range(len(extrinsics))])

    os.chdir("..")
    # frames_dir = f"{scene_name}/images"
    for type in split_types:
        print("===== Split type: " + type + " ===================")
        scene_dir = f"{scene_name}_{type}/"
        create_ace_dirs(scene_dir)

        split_fn = SCENE_SPLIT_FNS[SCENE_SPLIT_TYPES.index(type)]
        convert_ue_data(extrinsics, intrinsics, image_paths, scene_dir, split_fn)

    os.system("rm -rf " + scene_name)
