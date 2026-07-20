#!/bin/bash

set -e

###############################################################################
# Configuration
###############################################################################

MODEL="/Users/telikepalli/code-llm-distillation-thesis/lora/artifacts/deepseek-coder-1.3b-instruct-mlx"

DATA="/Users/telikepalli/code-llm-distillation-thesis/optimization/lora_datasets"

EXPERIMENT_NAME="LoRA Rank + Gradient Checkpoint"

BATCH_SIZE=1
ITERS=300
LEARNING_RATE=1e-5

STEPS_PER_REPORT=10
STEPS_PER_EVAL=20
SAVE_EVERY=50

###############################################################################
# Experiment Dimensions
###############################################################################

LORA_RANKS=(4 8 16 32)
GRADIENT_CHECKPOINTS=(false true)

###############################################################################
# Run Experiments
###############################################################################

for LORA_RANK in "${LORA_RANKS[@]}"
do
    for GC in "${GRADIENT_CHECKPOINTS[@]}"
    do

        RUN_NAME="lora_r${LORA_RANK}_gc_${GC}"

        ADAPTER_PATH="optimization/adapters/${RUN_NAME}"

        echo
        echo "============================================================"
        echo "Running ${RUN_NAME}"
        echo "============================================================"
        echo

        CMD=(
            python optimization/train_lora_pass5_mlx.py
            --experiment-name "${EXPERIMENT_NAME}"
            --model "${MODEL}"
            --data "${DATA}"
            --adapter-path "${ADAPTER_PATH}"
            --lora-rank "${LORA_RANK}"
            --batch-size "${BATCH_SIZE}"
            --iters "${ITERS}"
            --learning-rate "${LEARNING_RATE}"
            --steps-per-report "${STEPS_PER_REPORT}"
            --steps-per-eval "${STEPS_PER_EVAL}"
            --save-every "${SAVE_EVERY}"
        )

        if [ "${GC}" = "true" ]; then
            CMD+=(--grad-checkpoint)
        fi

        "${CMD[@]}"

    done
done

echo
echo "============================================================"
echo "All experiments completed."
echo "============================================================"