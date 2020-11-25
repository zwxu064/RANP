module load tensorboardx/1.2.0-py36-cuda90 torchvision/0.2.1-py36
module load intel-mkl/2017.2.174 intel-cc/17.0.2.174 intel-fc/17.0.2.174 caffe/0.16.6
module load pytorch/1.1.0-py36-cuda90 cuda/9.2.88 gcc/6.4.0 eigen/3.2.9 python/3.6.1 xfce/4.12

CUDA_VISIBLE_DEVICES="3" python train_brats.py \
--dataset="brats" \
--data_dir="datasets/BraTS"  \
--resume_epoch=-1 \
--test_epoch=-1 \
--optimizer="adam" \
--spatial_size=128 \
--prune_spatial_size=96 \
--enable_train \
--enable_cuda \
--width 2 \
--number_of_fmaps 4 \
--neuron_sparsity=0.7817 \
--resource_list_type="grad_flops"

#--enable_layer_neuron_display \
