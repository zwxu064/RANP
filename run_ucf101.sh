#!/bin/bash

#SBATCH --job-name=ucf101_mb
#SBATCH --time=2:00:00
#SBATCH --mem=17GB
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=logs/test_i3d.txt

# module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
# module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
# module load pytorch/1.1.0-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1 xfce/4.12

#python train_ucf101.py \
#--dataset="ucf101" \
#--video_path "./datasets/UCF101/images" \
#--annotation_path "./datasets/annotations/annotation_UCF101/ucf101_01.json" \
#--checkpoint_dir "results/ucf101/debug" \
#--model "mobilenetv2" \
#--groups 3 \
#--width_mult 1.0 \
#--downsample 1 \
#--n_threads 16 \
#--checkpoint 1 \
#--n_val_samples 3 \
#--resource_list_type "grad_flops" \
#--neuron_sparsity 0.3315 \
#--enable_test \
#--resume_path="models/ucf101_mobilenetv2/RANP_f/ucf101_mobilenetv2_1.0x_RGB_16_best.pth"

# --enable_test \
# --resume_path="models/ucf101_mobilenetv2/RANP_f/ucf101_mobilenetv2_1.0x_RGB_16_best.pth"
# --resource_list_type "grad_flops" \
# --neuron_sparsity 0.3315 \

CUDA_VISIBLE_DEVICES="2,3" \
python train_ucf101.py \
--dataset="ucf101" \
--video_path "./datasets/UCF101/images" \
--annotation_path "./datasets/annotations/annotation_UCF101/ucf101_01.json" \
--checkpoint_dir "results/ucf101/0194_grad_flops_80" \
--model "i3d" \
--groups 3 \
--width_mult 1.0 \
--downsample 1 \
--n_threads 32 \
--checkpoint 1 \
--n_val_samples 3 \
--resource_list_type "grad_flops" \
--neuron_sparsity 0.0194 \
--enable_train \
--pretrain_path="/home/users/u5710355/WorkSpace/gitlab/pytorch-projects/RANP/models/ucf101_i3d/rgb_imagenet.pt" \
--ft_portion="complete" \
--resource_list_lambda 80 \
--batch 32
#
## --enable_test \
## --resume_path="models/ucf101_i3d/RANP_f/ucf101_I3D_1.0x_RGB_16_best.pth"
## --resource_list_type "grad_flops" \
## --neuron_sparsity 0.2532 \
