
# Crowd counting: Classification-Based Technique

Crowd counting and crowd analysis are essential research areas in computer vision and image processing. They involve estimating the number of people in a crowd and analysing their behaviour, movement patterns, and interactions. This field has seen significant advancements in recent years due to its broad applications in various domains, such as crowd management, surveillance, urban planning, and event organization.


## Overview

![densities](https://github.com/Pshubham1012/Classification-approach/assets/124425044/ca18705a-8db4-4bfd-a1ff-fe2b64d25719)

## Network architecture
The overall architecture of the proposed classification net mainly consists of two components: DM-count model for crowd counting and Resnet_18 for classification. And it has been executed in three stages i) The initial pretraining stage ii) The classifier training stage iii) The final stage.

**i) Initial pretraining stage:** 
These three separate DM-count models are trained on high, low, and medium-density crowd data.

<img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/st1.png" alt="Image" style="width: 70%; height: 70%;">

**ii) Classifier training:**
In this, we segregated the crowd data into three classes using the three models pertained in the previous stage and then trained the Resnet_18 Classifier model on it.

<img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/st2.png" alt="Image" style="width: 70%; height: 70%;">

**iii)Final stage:**
In this we use all models pre-trained in stages i and ii to get the final prediction on the unknown image.

<img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/st3.png" alt="Image" style="width: 90%; height: 90%;">

The **DM-count** model used is the most simple and state-of-the-art crowd-counting model, shown below:

<img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/dm count.png" alt="Image" style="width: 70%; height: 70%;">

It consists of a VGG-16 backbone, pre-trained on Imagenet data and a regression head consisting of three convolution layers which give the density map. the overall network is finetuned on labelled crowd-counting data.
## Prerequisites

Python 3.x

Pytorch >= 1.2


## Getting Started
Dataset download

+ JHU-count can be downloaded [here](http://www.crowd-counting.com/#download)

**Stage1:**
1. Data directory structure
Place the dataset in the `../data/` folder. So the directory structure should look like the following:
```
-- data
   --ST_partA
     -- test_data
      -- ground-truth
      -- images
     -- train_data
      -- ground-truth
      -- images
```

2. Data preprocess

Due to the large sizes of images in the QNRF and NWPU datasets, we preprocess these two datasets.

```
python preprocess_dataset.py --dataset <dataset name: qnrf or nwpu> --input-dataset-path <original data directory> --output-dataset-path <processed data directory> 
```

3. Training

```
python train.py --dataset <dataset name: qnrf, sha, shb or nwpu> --data-dir <path to dataset> --device <gpu device id>
```

4. Test

```
python test.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset> --dataset <dataset name: qnrf, sha, shb or nwpu>
```
**do the same for stage 2 and stage 3**
