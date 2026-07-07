#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=logs/train_stage_1.log
#SBATCH -w veuc13


# Single frame training
PYOPENGL_PLATFORM=osmesa \
    PYTHONPATH=/home/usuaris/veu/cescola/hamer:/home/usuaris/veu/cescola/signvip/scripts/sk:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0,1 \
    accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 1 --gpu_ids "0,1" \
    train_stage1_multi_cond.py --config "configs/stage1/stage_1_multicond_RWTH.yaml"
    
