# Publication
- This work, [**RANP**](https://arxiv.org/abs/2010.02488), is accepted as an oral paper by 3DV 2020
  and awarded "Best Student Paper".
  If you find our paper or code useful, please cite it below.
  ```
  @inproceedings{xu:3dv2020ranp,
      title={RANP: Resource Aware Neuron Pruning at Initialization for 3D CNNs},
      author={Zhiwei Xu, Thalaiyasingam Ajanthan, and Richard Hartley},
      booktitle={International Conference on 3D Vision},
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
  conda install -c anaconda pytables==3.4.4
  conda install -c anaconda opencv==3.4.2
 
- A possible error when installing cv2 for UCF101 experiments is
  ```
  ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.22' not found
  ```
  To check if this version is in the conda RANP environment, run
  ```
  cd [path]/anaconda3/envs/RANP/lib
  strings libstdc++.so.6 | grep GLIBCXX
  ```
  If yes, add the following to ~/.bashrc, then source .bashrc;
  otherwise, download it, say [libstdc++.so.6.0.22](https://1drv.ms/u/s!AngC1-tRlyPMggac1z50VX9bB6cr),
  first, and replace libstdc++.so.6 with
  a new soft-link to this downloaded dynamic library.
  ```
  export LD_PRELOAD=[path]/anaconda3/envs/RANP/lib/libstdc++.so.6:$LD_PRELOAD
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
