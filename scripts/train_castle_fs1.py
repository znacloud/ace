import os
import subprocess
import pandas as pd
from types import SimpleNamespace

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

def read_pose_data(file_path):
  """Reads data from a file and returns a Pandas DataFrame."""
  column_names = ["file_name", "rot_quaternion_w", "rot_quaternion_x", "rot_quaternion_y", "rot_quaternion_z",
                  "translation_x", "translation_y", "translation_z", "rot_err_deg", "tr_err_m", "inlier_count"]
  df = pd.read_csv(file_path, delim_whitespace=True, names=column_names)
  return df

# Find the path to the root of the repo.
script_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.realpath(os.path.join(script_path, ".."))

test_scene = "castle_fs_ts"
train_scene = "castle_fs_n1"
train_n = 1
image_names_in_train = ["image_fs_r5000_s1_00000.jpeg"]
# scenes = ["castle_fs_ts",]

training_exe = os.path.join(repo_path, "train_ace.py")
testing_exe = os.path.join(repo_path, "test_ace.py")

datasets_folder = os.path.join(repo_path, "datasets")
out_dir = os.path.join(repo_path, "output/castle_fs")
os.makedirs(out_dir, exist_ok=True)

# Run training and testing for each scene
while train_scene != None:
    print(f"Training on scene: {train_scene}")
    train_command = ["python", training_exe, os.path.join(datasets_folder, train_scene), os.path.join(out_dir, f"{train_scene}.pt")]
    test_command = ["python", testing_exe, os.path.join(datasets_folder, test_scene), os.path.join(out_dir, f"{train_scene}.pt")]

    # Execute training
    subprocess.run(train_command, check=True)

    # Execute testing and log output
    log_file_path = os.path.join(out_dir, f"log_{train_scene}.txt")
    with open(log_file_path, "w") as log_file:
        subprocess.run(test_command, stdout=log_file, stderr=subprocess.STDOUT, check=True)

    with open(log_file_path, "r") as log_file:
        log_lines = log_file.readlines()
        if len(log_lines) >= 5:
            print(f"{train_scene}: {log_lines[-5].strip()}")

    # Read pose file and check the errors
    pose_file_path = os.path.join(out_dir, f"poses_{test_scene}_.txt")
    print(f"Read pose file {pose_file_path}")
    pose_dp = read_pose_data(pose_file_path)
    
    # Filter out the rows where image_name is already in image_names_in_train array
    pose_dp = pose_dp[~pose_dp["file_name"].isin(image_names_in_train)]
    print(f"Filtered out pose file already in train dataset, remaining rows: {len(pose_dp)}")

    # keep only the rows where the rotation error is greater than 5 degrees
    # or translation error is greater than 0.05 meters
    pose_dp = pose_dp[(pose_dp["rot_err_deg"] > 5) | (pose_dp["tr_err_m"] > 0.05)]
    print(f"Filtered out pose file > 5deg/5cm, remaining rows: {len(pose_dp)}")

    # Sort by translation error descending then by rotation error desending
    pose_dp = pose_dp.sort_values(by=["tr_err_m", "rot_err_deg"], ascending=[False, False])

    # Keep only the first 10 rows
    pose_dp = pose_dp.head(10)
    delta = min(10, len(pose_dp))
    print(f"Keep at most 10 rows, actual rows: {delta}")
    if delta > 0:
        train_n += delta
        print(f"Prepare next train scene: castle_fs_n{train_n}")
        next_train_scene = f"castle_fs_n{train_n}"
        next_train_dir = os.path.join(datasets_folder, next_train_scene)
        # Copy the whole train_scene folder to next_train_scene folder
        print(f"Copy {train_scene} to {next_train_scene}")
        subprocess.run(["cp", "-r", os.path.join(datasets_folder, train_scene), next_train_dir], check=True)

        # For all image_name in pose_dp, copy the corresponding image to next_train_scene folder
        print(f"Copy images in pose_dp to {next_train_scene}")
        for image_name in pose_dp["file_name"]:
            subprocess.run(["cp", os.path.join(datasets_folder, test_scene, ACE_DIRS.test.rgb, image_name),
                            os.path.join(next_train_dir, ACE_DIRS.train.rgb)], check=True)
            subprocess.run(["cp", os.path.join(datasets_folder, test_scene, ACE_DIRS.test.calib, image_name[:-4] + "txt"),
                            os.path.join(next_train_dir, ACE_DIRS.train.calib)], check=True)
            subprocess.run(["cp", os.path.join(datasets_folder, test_scene, ACE_DIRS.test.pose, image_name[:-4] + "txt"),
                            os.path.join(next_train_dir, ACE_DIRS.train.pose)], check=True)

        train_scene = next_train_scene 

    

