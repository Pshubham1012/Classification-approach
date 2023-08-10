
# Crowd counting: Classification based Technique

Crowd counting and crowd analysis are essential research areas in computer vision and image processing. They involve estimating the number of people in a crowd and analysing their behaviour, movement patterns, and interactions. This field has seen significant advancements in recent years due to its broad applications in various domains, such as crowd management, surveillance, urban planning, and event organization.


## Overview

![densities](https://github.com/Pshubham1012/Classification-approach/assets/124425044/ca18705a-8db4-4bfd-a1ff-fe2b64d25719)

## Network architecture
The overall architecture of the proposed classification net mainly consists of two components: DM-count model for crowd counting and Resnet_18 for classsification. And it has been executed in three stages i) Initial pretraining stage ii) Classifier training stage iii) Final stage.

**i) Initial pretraining stage:** 
In this three seperate DM-count models are trained on high, low, and medium density crowd data.
![image](https://github.com/Pshubham1012/Classification-approach/assets/124425044/6a276e5b-648e-4150-81a1-d901baec8f13)
**ii) Classifier training:**
In this we segregated the crowd data in three classes using the three models pretained in previous stage and then train the Resnet_18 Classifier model on it.
![image](https://github.com/Pshubham1012/Classification-approach/assets/124425044/691b00e2-3db7-4607-a0a3-2f89c4fd2e29)
**iii)Final stage:**
In this we use all models pretrained in stage i and ii to get the final prediction on unknown image.
![image](https://github.com/Pshubham1012/Classification-approach/assets/124425044/89337109-259a-4841-8de9-d734384b9e95)

The **DM-count** model used is the most simple and state of art crowd counting model, shown below:
![dm count](https://github.com/Pshubham1012/Classification-approach/assets/124425044/0e4585c1-474a-4b58-ade4-aadadb77a14d)

<img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/dm count.png" alt="Image" style="width: 50%; height: 50%;">

It is consist of VGG-16 backbone, pretrained on Imagenet data and regression head consist of threee convolution layers which gives the density map. overall network is finetuned on labelled crowd counting data.
## Prerequisites

Python 3.x

Pytorch >= 1.2


## Getting Started
Dataset download

+ JHU-count can be downloaded [here](http://www.crowd-counting.com/#download)

**Stage1:**
1. Data directory structure
Place the dataset in `../data/` folder. So the directory structure should look like the following:
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

Due to large sizes of images in QNRF and NWPU datasets, we preprocess these two datasets.

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
**do the sam efor stage 2 and stage 3**