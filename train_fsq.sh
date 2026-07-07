#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs                                                            
#SBATCH --output=logs/train_stage_2.log
#SBATCH -w veuc13


# Single frame training
PYOPENGL_PLATFORM=osmesa \
    PYTHONPATH=/home/usuaris/veu/cescola/hamer:/home/usuaris/veu/cescola/signvip/scripts/sk:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0,1 \
    accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 2 --gpu_ids "0,1" \
    train_compress_vq_multicond.py \
    --config "configs/vq/vq_multicond_RWTH_compress.yaml"   
