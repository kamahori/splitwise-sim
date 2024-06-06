#!/bin/bash

# Activate conda environment
# source /home/yilegu/miniconda3/bin/activate splitwise

SCHEDULER=random_moe
START_STATE=mixtral
# TRACE=test_trace
TRACE=rr_code_30

python run.py \
    cluster=dgx-h100 \
    applications=mixtral \
    applications.0.scheduler=$SCHEDULER \
    start_state=$START_STATE \
    performance_model=moe \
    trace.filename=$TRACE \
    debug=True \
    seed=0
