#!/bin/bash
#SBATCH --job-name=psm
#SBATCH --time=72:00:00
#SBATCH --mem=17GB
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --output=./logs/train_psm.txt

# # module load python/2.7.13 cuda/9.0.176 pytorch/0.4.1-py27-cuda90
# module load torchvision/0.2.1-py36 python/3.6.1 cuda/9.0.176 pytorch/1.1.0-py36-cuda90

python train_stereo.py \
--dataset='SceneFlow' \
--maxdisp 192 \
--datapath ./datasets/SceneFlow/ \
--epochs 15 \
--savemodel ./trained/ \
--neuron_sparsity=0.462 \
--resource_list_type "grad_flops" \
--resource_list_lambda=100 \
--batch=12 \
--PSM_mode="max" \
--acc_mode="sum" \
--enable_raw_grad \
# --loadmodel ./ trained/checkpoint_6.tar \

# python finetune.py --maxdisp 192 \
#                    --model stackhourglass \
#                    --datatype 2015 \
#                    --datapath dataset/data_scene_flow_2015/training/ \
#                    --epochs 300 \
#                    --loadmodel ./trained/checkpoint_10.tar \
#                    --savemodel ./trained/

