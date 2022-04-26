#!/bin/bash
set -x
# https://stackoverflow.com/questions/34709749/how-do-i-use-nvidia-multi-process-service-mps-to-run-multiple-non-mpi-cuda-app
# the following must be performed with root privilege

GPU_ID=$1

# export CUDA_VISIBLE_DEVICES=$GPU_ID
# nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS
# nvidia-cuda-mps-control -d

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=1,2
for GPU_ID in $(seq 0 3); do
    nvidia-smi -i $GPU_ID -c EXCLUSIVE_PROCESS
done
taskset 0x00000001  nvidia-cuda-mps-control -d
