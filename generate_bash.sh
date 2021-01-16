#!/bin/bash -l

# Config
dataset="sceneflow"
enable_debug=true
enable_run_script=true
resource_list_type="grad_flops"
resource_list_lambda=0
sparsity=0.7  # max: 0.90014
mode="min"
acc_mode="sum"
prune_mode="neuron"
cnn_mode="test"
enable_raw_grad=false

if [ ${cnn_mode} == "train" ]; then
    cluster_time=72
else
    cluster_time=10
fi

if [ ${prune_mode} != "neuron" ]; then
    resource_list_type="vanilla"
    resource_list_lambda=0
    num_gpus=3
else
    if [ `echo "${sparsity} > 0.5"|bc` -eq 1 ]; then
        num_gpus=2
    else
        num_gpus=3
    fi
fi

if [ ${cnn_mode} == "test" ]; then
    num_gpus_train=${num_gpus}
    num_gpus=1
fi

common="${dataset}_${prune_mode}_${cnn_mode}_${resource_list_type}_lambda${resource_list_lambda}_spa${sparsity}_${mode}_gpus${num_gpus}_accmode${acc_mode}"
loadmodel="checkpoints/stereo/${dataset}_${prune_mode}_train_${resource_list_type}_lambda${resource_list_lambda}_spa${sparsity}_${mode}_gpus${num_gpus_train}_accmode${acc_mode}"

if ${enable_raw_grad}; then
    common+="_rawgrad"
    loadmodel+="_rawgrad"
fi

bash_name="scripts/stereo/${common}.sh"
log_name="logs/stereo/${common}.txt"
result_path="checkpoints/stereo/${common}"

if ${enable_debug}; then
    log_name="logs/stereo/${common}_debug.txt"
    result_path="checkpoints/debug_stereo"
    cluster_time=1
fi

echo ${log_name}

echo -e "#!/bin/bash -l

#SBATCH --job-name=${mode}_${acc_mode}_${sparsity}
#SBATCH --time=${cluster_time}:00:00
#SBATCH --mem=17G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:${num_gpus}
#SBATCH --mail-user=
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=${log_name}
" > ${bash_name}

# Write modules
echo -e "module load torchvision/0.2.1-py36 python/3.6.1 cuda/9.0.176 pytorch/1.1.0-py36-cuda90" >> ${bash_name}

# Write main command
echo -e "
python train_stereo.py \\
--dataset=\"SceneFlow\" \\
--maxdisp=192 \\
--datapath=\"./datasets/SceneFlow/\" \\
--epochs=15 \\
--acc_mode=\"${acc_mode}\" \\
--PSM_mode=\"${mode}\" \\" >> ${bash_name}

if [ ${prune_mode} == "neuron" ]; then
    echo -e "--neuron_sparsity=${sparsity} \\
--resource_list_type=\"${resource_list_type}\" \\
--resource_list_lambda=${resource_list_lambda} \\" >> ${bash_name}
elif [ ${prune_mode} == "param" ]; then
    echo -e "--param_sparsity=${sparsity} \\" >> ${bash_name}
elif [ ${prune_mode} == "layerwise" ]; then
    echo -e "--layer_sparsity_list=${sparsity} \\" >> ${bash_name}
elif [ ${prune_mode} == "random" ]; then
    echo -e "--random_sparsity=${sparsity} \\
--random_method=0 \\" >> ${bash_name}
fi

if [ ${cnn_mode} == "train" ]; then
    echo -e "--enable_train \\
--batch=12 \\
--savemodel=\"${result_path}\" \\" >> ${bash_name}
elif [ ${cnn_mode} == "test" ]; then
    echo -e "--enable_test \\
--batch=8 \\
--loadmodel=\"${loadmodel}\" \\" >> ${bash_name}
fi

if ${enable_raw_grad}; then
    echo -e "--enable_raw_grad \\" >> ${bash_name}
fi

eval "chmod 755 ${bash_name}"

# Run
if ${enable_run_script}; then
    eval "sbatch ${bash_name}"
fi
