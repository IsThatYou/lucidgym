#!/usr/bin/env bash

set -euo pipefail
source ../../.venv/bin/activate

# Basic knobs (adjust directly in this file or pass as args):
DATASET_NAME="arcagi3-as66"
ROOT_URL="https://three.arcprize.org"

# Default train/test splits with simple repetition counts.
# train_games=(
#   "as66-821a4dcad9c2"
#   "ls20-fa137e247ce6"
#   "ft09-b8377d4b7815"
#   "vc33-6ae7bf49eea5"
#   "lp85-d265526edbaa"
#   "sp80-0605ab9e5b2a"
# )

train_games="as66-821a4dcad9c2"
train_reps=500

test_games="as66-821a4dcad9c2"
test_reps=1

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python "${script_dir}/prepare_arc_agi_data.py" \
  --dataset-name "${DATASET_NAME}" \
  --root-url "${ROOT_URL}" \
  --train-game-ids "${train_games}" \
  --train-repetitions "${train_reps}" \
  --test-game-ids "${test_games}" \
  --test-repetitions "${test_reps}"
