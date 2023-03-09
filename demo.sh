#!/bin/bash

patch -Nu ../pfrl/pfrl/agents/dqn.py < ./docker/patch/dqn.py.patch

model=Action_Q_Transformer_model
load="results/Action_Q_Transformer_model/20230213T171349.727168/best"

# Start tuning hyperparameters
python tegaki_train_franka_dqn.py \
    --outdir results/demo/${model} \
    --model ${model} \
    --num-envs 12 \
    --num-items 1 \
    --item-names item21 \
    --target item21 \
    --isaacgym-assets-dir /opt/isaacgym/assets \
    --item-urdf-dir ./urdf \
    --load ${load} \
    --gpu 0 \
    --descentstep 12 \
    --demo \
    --hand \
    --render \

