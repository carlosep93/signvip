#!/bin/bash


#SBATCH -p veu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=50G # Memory
#SBATCH --ignore-pbs
#SBATCH --output=logs/run_hamer_preprocessing.log
#SBATCH -w veuc11

 export PYTHONPATH=/home/usuaris/veu/cescola/hamer:/home/usuaris/veu/cescola/signvip/scripts/sk:$PYTHONPATH

path_to_videos='/home/usuaris/veussd/carlos.escolano/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/train_processed_videos'
path_to_sk='/home/usuaris/veussd/carlos.escolano/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/train_processed_videos/sk'
path_to_hamer='/home/usuaris/veussd/carlos.escolano/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/train_processed_videos/hamer'
path_to_hamer_rendered='/home/usuaris/veussd/carlos.escolano/phoenix/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/train_processed_videos/hamer_rendered'


# PYOPENGL_PLATFORM=osmesa \
#    PYTHONPATH=/home/usuaris/veu/cescola/hamer:/home/usuaris/veu/cescola/signvip/scripts/sk:$PYTHONPATH \
#    CUDA_VISIBLE_DEVICES=0 python scripts/hamer/infer_videos_dw.py \
#    --video_folder $path_to_videos \
#    --pose_folder $path_to_sk \
#    --out_folder $path_to_hamer



cd /home/usuaris/veu/cescola/hamer
PYTHONPATH=/home/usuaris/veu/cescola/hamer:/home/usuaris/veu/cescola/signvip/scripts/sk:$PYTHONPATH \
    CUDA_VISIBLE_DEVICES=0 \
    python /home/usuaris/veu/cescola/signvip/scripts/hamer/save_video.py \
    --faces_path /home/usuaris/veu/cescola/hamer/faces.npz \
    --video_folder $path_to_videos \
    --pose_folder  $path_to_hamer \
    --out_folder   $path_to_hamer_rendered
