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
    python analyze_fsq_distribution.py \
    --config configs/vq/vq_multicond_RWTH_compress.yaml \
    --checkpoint workspace/vq_multicond_RWTH_compress/20260708-1100/best/condition_encoder/model.bin \
    --split val \
    --output_dir fsq_analysis
