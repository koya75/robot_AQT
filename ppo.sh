#!/bin/bash

model=handwritten_instructions_transformer  # mask_double , handwritten_instructions_transformer
robot=35

# Start tuning hyperparameters
python -B hwi_train_franka_ppo.py \
    --outdir results/${model}_lstm_hoge \
    --model ${model} \
    --epochs 10 \
    --gamma 0.99 \
    --step_offset 0 \
    --lambd 0.995 \
    --lr 0.0002 \
    --max-grad-norm 40 \
    --step_offset 0 \
    --gpu 1 \
    --num-envs ${robot} \
    --eval-n-runs ${robot} \
    --update-batch-interval 1 \
    --num-items 2 \
    --item-names item21 item25\
    --isaacgym-assets-dir /opt/isaacgym/assets \
    --item-urdf-dir ./urdf \
    --steps 25000000 \
    --eval-batch-interval 20 \
    --descentstep 40 \
    # --use-lstm \
    # --render

#25000000
#results
#--hand \
#--mode normal \hard
#    --load results/${model}_lstm/20220622T182921.228762/9581440_except \
#    --use-lstm \--item-names item25 item38 item21 item22 item23 item24 \
#    --eval-n-runs ${robot} \