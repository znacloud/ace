#!/usr/bin/env bash

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

scenes=("castle_seq1")

training_exe="${REPO_PATH}/train_ace.py"
testing_exe="${REPO_PATH}/test_ace.py"

datasets_folder="${REPO_PATH}/datasets"
out_dir="${REPO_PATH}/output/viz_maps/castle"
renderings_train_dir="${REPO_PATH}/output/renderings/castle_train"
renderings_test_dir="${REPO_PATH}/output/renderings/castle_test"

mkdir -p "$out_dir"
mkdir -p "$renderings_train_dir"
mkdir -p "$renderings_test_dir"

for scene in ${scenes[*]}; do
  python $training_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True --render_target_path "$renderings_train_dir"
  ffmpeg -framerate 30 -pattern_type glob -i "$renderings_train_dir/$scene/*.png" -c:v libx264 -pix_fmt yuv420p "$renderings_train_dir/$scene.mp4"
  python $testing_exe "$datasets_folder/$scene" "$out_dir/$scene.pt" --render_visualization True  --render_target_path "$renderings_test_dir"
  ffmpeg -framerate 30 -pattern_type glob -i "$renderings_test_dir/$scene/*.png" -c:v libx264 -pix_fmt yuv420p "$renderings_test_dir/$scene.mp4"
done