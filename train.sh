#!/bin/bash
model=dqn  #dqn, Action_Q_Transformer_model, mask_double
robot=70

# Start tuning hyperparameters
python -B train_franka.py \
    --outdir results/${model} \
    --model ${model} \
    --epochs 10 \
    --gamma 0.97 \
    --step_offset 0 \
    --lambd 0.995 \
    --lr 0.0002 \
    --max-grad-norm 40 \
    --step_offset 0 \
    --gpu 1 \
    --num-envs ${robot} \
    --eval-n-runs ${robot} \
    --update-batch-interval 1 \
    --num-items 3 \
    --item-names item25 item21 item38 \
    --target item21 \
    --isaacgym-assets-dir /opt/isaacgym/assets \
    --item-urdf-dir ./urdf \
    --steps 100000000 \
    --eval-batch-interval 20 \
    --descentstep 12 \
    --hand \
    --render
