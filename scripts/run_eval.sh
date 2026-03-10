#!/bin/bash

# OpenVLA Batch Evaluation Script
# Run evaluation on all LIBERO task suites

set -e

# Configuration
MODEL_BASE_PATH="/data1"  # Adjust to your model weights directory
NUM_TRIALS=50             # Number of trials per task (default: 50)
CENTER_CROP=True          # Use center crop during inference

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "OpenVLA LIBERO Batch Evaluation"
echo "=========================================="

# Task suites to evaluate
TASK_SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

for suite in "${TASK_SUITES[@]}"; do
    echo -e "\n${YELLOW}Evaluating task suite: $suite${NC}"

    MODEL_PATH="${MODEL_BASE_PATH}/openvla-7b-finetuned-${suite}"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Model not found at $MODEL_PATH, skipping..."
        continue
    fi

    python experiments/robot/libero/run_libero_eval.py \
        --model_family openvla \
        --pretrained_checkpoint "$MODEL_PATH" \
        --task_suite_name "$suite" \
        --center_crop "$CENTER_CROP" \
        --num_trials_per_task "$NUM_TRIALS"

    echo -e "${GREEN}✓ Completed evaluation for $suite${NC}"
done

echo -e "\n${GREEN}=========================================="
echo "All evaluations completed!"
echo "==========================================${NC}"
echo "Check ./rollouts/ for generated videos"
