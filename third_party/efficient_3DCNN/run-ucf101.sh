#!/bin/bash -l

#SBATCH --job-name=train_ucf101
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mail-user=zhiwei.xu@anu.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=results/bash_logs/train_ucf101_4GPUs.txt

module load pytorch/1.0.1-py36-cuda90 ffmpeg/3.2.4 opencv/3.4.3

python main.py \
--root_path ./ \
--video_path ./datasets/ucf101/images \
--annotation_path ./annotation_UCF101/ucf101_01.json \
--result_path ./results/running/4GPUs \
--dataset ucf101 \
--n_classes 101 \
--model mobilenetv2 \
--groups 3 \
--width_mult 1.0 \
--train_crop random \
--learning_rate 0.1 \
--sample_duration 16 \
--downsample 1 \
--batch_size 64 \
--n_threads 16 \
--checkpoint 1 \
--n_val_samples 1 \