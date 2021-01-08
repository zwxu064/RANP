#!/bin/bash

# Copyright 2020-present, zhiwei.xu@anu.edu.au,
# All rights reserved.

# For ShapeNet, this data processing is based on
# [Spatially-sparse convolutional networks](https://github.com/btgraham/SparseConvNet)
# if you use this code, you should cite "[Sparse 3D convolutional neural networks](http://arxiv.org/abs/1505.02890)"

# Download  https://shapenet.cs.stanford.edu/iccv17/ competition data.
# We re-split the Train/Validation data 50-50 to increase the size of the validation set.

if [[ ! -f "ShapeNet" ]]; then
    mkdir ShapeNet
fi

cd ShapeNet

if [[ ! -f "train_data" ]]; then
    wget https://shapenet.cs.stanford.edu/iccv17/partseg/train_data.zip
    wget https://shapenet.cs.stanford.edu/iccv17/partseg/train_label.zip
    wget https://shapenet.cs.stanford.edu/iccv17/partseg/val_data.zip
    wget https://shapenet.cs.stanford.edu/iccv17/partseg/val_label.zip
    wget https://shapenet.cs.stanford.edu/iccv17/partseg/test_data.zip
    wget https://shapenet.cs.stanford.edu/iccv17/partseg/test_label.zip
    unzip train_data.zip
    unzip train_label.zip
    unzip val_data
    unzip val_label
    unzip test_data.zip
    unzip test_label.zip
fi

for x in train_val test; do 
    for y in 02691156 02773838 02954340 02958343 03001627 03261776 03467517 03624134 03636649 03642806 03790512 03797390 03948459 04099429 04225987 04379243; do 
        mkdir -p $x/$y
    done
done

for x in 02691156 02773838 02954340 02958343 03001627 03261776 03467517 03624134 03636649 03642806 03790512 03797390 03948459 04099429 04225987 04379243; do 
    mv train_*/$x/* val_*/$x/* train_val/$x/; cp test_*/$x/* test/$x/
done

rm -rf train_data train_label val_data val_label test_data test_label

for x in train_val/*/*.pts; do 
    y=`md5sum $x|cut -c 1|tr -d 89abcdef`
    if [ $y ]; then 
        mv $x $x.train
    else 
        mv $x $x.valid
    fi
done

cd ..