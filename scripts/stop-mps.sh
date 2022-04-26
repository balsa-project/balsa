#!/bin/bash
GPU_ID=$1
echo quit | nvidia-cuda-mps-control
nvidia-smi -i $GPU_ID -c DEFAULT
