#!/bin/bash

# Activate conda environment
source /home/yilegu/miniconda3/bin/activate splitwise

SCHEDULER=random_moe
START_STATE=mixtral
TRACE=test_trace

python run.py \
    cluster=dgx-h100 \
    applications.0.scheduler=$SCHEDULER \
    start_state=$START_STATE \
    performance_model=db \
    trace.filename=$TRACE \
    debug=True \
    seed=0
