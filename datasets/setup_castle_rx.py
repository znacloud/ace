#!/usr/bin/env python3

import os
from dataclasses import dataclass
import numpy as np
import shutil
import dataset_util as dutil
from types import SimpleNamespace
from scipy.spatial.transform import Rotation


scene_url = "https://sagemaker-studio-163033260074-0o5wme941ix9.s3.ap-southeast-2.amazonaws.com/castle_rx.zip"
scene_file = "castle_rx.zip"
scene_name = "castle_rx"

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


# test set: curve_path_r5000_1 + curve_path_r5000_2
# t_base - baseline train: curve_path_r5000_3 + cp_r5000_4 + cp_r5000_5
# t_s1: cp_r4500_3 + cp_r5000_4 + cp_r5500_5
# t_s2: fs_poses_r5000_s2 + fs_poses_r5000_s4
# t_s3: fs_poses_r4500_s1 + fs_poses_r5500_s3
# t_s4: fs_poses_r5000_sx (20-30m height)
# t_s5: fs_poses_r4500_s1 + fs_poses_r5000_s2 + fs_poses_r5000_s4 + fs_poses_r5500_s3 (20-30m height)

def train_test_split_base(idx, image_name, total, pose=[0,0,0]):
    if image_name.startswith("image_r5000_c1") or image_name.startswith("image_r5000_c2"):
        return ACE_DIRS.test
    elif image_name.startswith("image_r5000_c3") or image_name.startswith("image_r5000_c4") or image_name.startswith("image_r5000_c5") :
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_1(idx, image_name, total, pose=[0,0,0]):
    if image_name.startswith("image_r5000_c1") or image_name.startswith("image_r5000_c2"):
        return ACE_DIRS.test
    elif image_name.startswith("image_r4500_c3") or image_name.startswith("image_r5000_c4") or image_name.startswith("image_r5500_c5") :
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_2(idx, image_name, total, pose=[0,0,0]):
    if image_name.startswith("image_r5000_c1") or image_name.startswith("image_r5000_c2"):
        return ACE_DIRS.test
    elif image_name.startswith("image_fs_r5000_s2") or image_name.startswith("image_fs_r5000_s4"):
        return ACE_DIRS.train
    else:
        return False

def train_test_split_3(idx, image_name, total, pose=[0,0,0]):
    if image_name.startswith("image_r5000_c1") or image_name.startswith("image_r5000_c2"):
        return ACE_DIRS.test
    elif image_name.startswith("image_fs_r4500_s1") or image_name.startswith("image_fs_r5500_s3"):
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_4(idx, image_name, total, pose=[0,0,0]):
    if image_name.startswith("image_r5000_c1") or image_name.startswith("image_r5000_c2"):
        return ACE_DIRS.test
    elif image_name.startswith("image_fs_r5000_") and pose[2] >= 20 and pose[2] <=30: #meters
        return ACE_DIRS.train
    else:
        return False
    
def train_test_split_5(idx, image_name, total, pose=[0,0,0]):
    if image_name.startswith("image_r5000_c1") or image_name.startswith("image_r5000_c2"):
        return ACE_DIRS.test
    elif (image_name.startswith("image_fs_r4500_s1") or image_name.startswith("image_fs_r5000_s2") \
        or image_name.startswith("image_fs_r5000_s4") or image_name.startswith("image_fs_r5500_s3")) \
        and pose[2] >= 20 and pose[2] <=30: #meters
        return ACE_DIRS.train
    else:
        return False
    



SCENE_SPLIT_FNS = [
    train_test_split_base,
    train_test_split_1,
    train_test_split_2,
    train_test_split_3,
    train_test_split_4,
    train_test_split_5,
    
]
SCENE_SPLIT_TYPES = ["tbase","ts1","ts2","ts3","ts4","ts5"]


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
        mode_dir = mode_dir_fn(idx, image_file, len(cam_extrinsics), pose=[extr.x, extr.y, extr.z])
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
