#!/bin/bash
# Launch concurrent experiments of a config.
#    bash scripts/launch.sh <RUN> <N>
#
# Example usage:
#
#    # Launch 8 concurrent runs of a config 'RUN'.
#    RUN=Balsa_JOBRandSplit; bash scripts/launch.sh $RUN 8 2>&1 | tee launch-$RUN.log
#
#    # In a separate window, monitor a particular run.
#    # This will show query execution progress, training messages, etc.
#    tail -f $RUN-1.log
#
#    # Monitor all runs' mtime to check for any errored runs.
#    # Alternatively, look at W&B UI.
#    watch -n10 "ls -lthr ${RUN}-*.log launch-${RUN}.log"

RUN=$1
N=$2

# If desired, use this array to manully assign gpu ids.
gpus=(1 1 0 1 2 3 2 0)
numgpu=$(nvidia-smi --list-gpus | wc -l)

for i in $(seq 1 $N);
do
    # Mod.  OK for first launch.
    gpu_id=$(($(($i + $numgpu - 1)) % $numgpu))

    # Later launches: manually assign IDs to load balance.
    # gpu_id=${gpus[$(($i-1))]}

    # Use this to check.
    # echo $gpu_id

    CUDA_VISIBLE_DEVICES=$gpu_id python -u run.py --run $RUN 2>&1 >$RUN-$i.log &
    sleep 10
done

wait
