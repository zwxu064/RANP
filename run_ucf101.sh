module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
module load pytorch/1.1.0-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1 xfce/4.12

CUDA_VISIBLE_DEVICES="3" python train_ucf101.py \
--dataset="ucf101" \
--video_path "./datasets/UCF101/images" \
--annotation_path "./datasets/annotations/annotation_UCF101/ucf101_01.json" \
--checkpoint_dir "results/ucf101/debug" \
--model "mobilenetv2" \
--groups 3 \
--width_mult 1.0 \
--downsample 1 \
--n_threads 16 \
--checkpoint 1 \
--n_val_samples 3 \
--resource_list_type "grad_flops" \
--neuron_sparsity 0.3315

#CUDA_VISIBLE_DEVICES="3" python train_ucf101.py \
#--dataset="ucf101" \
#--video_path "./datasets/UCF101/images" \
#--annotation_path "./datasets/annotations/annotation_UCF101/ucf101_01.json" \
#--checkpoint_dir "results/ucf101/debug" \
#--model "i3d" \
#--groups 3 \
#--width_mult 1.0 \
#--downsample 1 \
#--n_threads 16 \
#--checkpoint 1 \
#--n_val_samples 3 \
#--resource_list_type "grad_flops" \
#--neuron_sparsity 0.2532