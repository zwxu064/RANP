# Related Publication
- This work is accepted as an oral paper by 3DV 2020. If you find our paper or code useful, please cite our work as follows.

    [**"RANP: Resource Aware Neuron Pruning at Initialization for 3D CNNs"**](https://arxiv.org/abs/2010.02488)\
    Zhiwei Xu, Thalaiyasingam Ajanthan, Vibhav Vineet, Richard Hartley\
    Internatinoal Conference on 3D Vision (3DV), November 2020, Japan (<span style="color:red">Oral</span>)

# Demo
- This repository will include demos of neuron pruning on 3D-UNets for 3D semantic segmentation as well as MobileNetV2 and I3D for video classification.

# How to use
- Step 1: download [precalculated gradients](https://1drv.ms/u/s!AngC1-tRlyPMgRKlb505D_db0RAO?e=zxHJT3) (users can also skip this step, then Step 2 will take time to generate these files automatically) from OneDrive. Then, generate folder "data" and put them in individual subfolders as "data/shapenet", "data/brats", and "data/ucf101".
- Step 2: run "./run_shapenet.sh" for ShapeNet experiments, "./run_brats.sh" for BraTS, and "./run_ucf101.sh" for UCF101 (set MobileNetV2 or I3D in this file)

# Notes
-  We will keep updating this repository, so if you have any questions, please contact zhiwei.xu@anu.edu.au. Thank you!
