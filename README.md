# Publication
- This work, [**RANP**](https://arxiv.org/abs/2010.02488), is accepted as an oral paper by 3DV 2020
  and awarded "Best Student Paper".
  If you find our paper or code useful, please cite it below.
  ```
  @article{xu2020ranp,
  title={RANP: Resource Aware Neuron Pruning at Initialization for 3D CNNs},
  author={Zhiwei Xu, Thalaiyasingam Ajanthan, Vibhav Vineet, and Richard Hartley},
  journal={Internatinoal Conference on 3D Vision},
  year={2020}
  }
  ```

# Demo
- This repository will include demos of neuron pruning on 3D-UNets for 3D semantic segmentation as well as MobileNetV2 and I3D for video classification.

# Environment
- Dependency
  ```
  conda create -n RANP python=3.6.12
  source activate RANP
  conda install pytorch=1.1.0 torchvision cudatoolkit=9.0 -c pytorch
  conda install -c conda-forge tensorboardx
  conda install -c anaconda scipy==1.3.2
  conda install -c conda-forge nibabel==3.2.1
  conda install -c conda-forge nilearn==0.7.0
  conda install -c anaconda pytables==3.6.1
  ```

- Datasets and proprocessing
  ```
  cd datasets
  ./download_datasets.sh
  ```

# How to use
- Create subfolders below.
    ```
    mkdir data data/shapenet data/brats data/ucf101
    ```
    Then, download [precalculated gradients](https://1drv.ms/u/s!AngC1-tRlyPMgRKlb505D_db0RAO?e=zxHJT3) from OneDrive
    (users can also skip this step, then running the following bash scripts will take time to generate these files automatically).
    Put them in the subfolders individually.
- Run: configurations in the bash files are default, change the argparse parameters carefully
when necessary.
  - For ShapeNet experiments,
    ```
    ./run_shapenet.sh
    ```
  - For BraTS experiments,
    ```
    ./run_brats.sh
    ```
  - For for UCF101 experiments (set MobileNetV2 or I3D in this file),
    ```
    ./run_ucf101.sh
    ```  

# Notes
-  We will keep updating this repository.
If you have any questions, please contact zhiwei.xu@anu.edu.au.