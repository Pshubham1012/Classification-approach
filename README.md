
# Crowd counting: Classification-Based Technique

Crowd counting and crowd analysis are essential research areas in computer vision and image processing. They involve estimating the number of people in a crowd and analysing their behaviour, movement patterns, and interactions. This field has seen significant advancements in recent years due to its broad applications in various domains, such as crowd management, surveillance, urban planning, and event organization.


## Overview

![densities](https://github.com/Pshubham1012/Classification-approach/assets/124425044/ca18705a-8db4-4bfd-a1ff-fe2b64d25719)

## Network architecture
The overall architecture of the proposed classification net mainly consists of two components: DM-count model for crowd counting and Resnet_18 for classification. And it has been executed in three stages i) The initial pretraining stage ii) The classifier training stage iii) The final stage.

**i) Initial pretraining stage:** 
These three separate DM-count models are trained on high, low, and medium-density crowd data.
<p align="center">
  <img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/st1.png" alt="Image" width="50%" height="50%">
</p>


   

**ii) Classifier training:**
In this, we segregated the crowd data into three classes using the three models pertained in the previous stage and then trained the Resnet_18 Classifier model on it.

<p align="center">
  <img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/st2.png" alt="Image" style="width: 50%; height: 50%;">
</p>

**iii)Final stage:**
In this we use all models pre-trained in stages i and ii to get the final prediction on the unknown image.

<p align="center">
  <img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/st3.png" alt="Image" style="width: 70%; height: 70%;">
</p>

The **DM-count** model used is the most simple and state-of-the-art crowd-counting model, shown below:

<p align="center">
  <img src="https://github.com/Pshubham1012/Classification-approach/raw/main/images/dm count.png" alt="Image" style="width: 50%; height: 50%;">
</p>

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
note: The JHU data has been categorized into three folders based on density: low, medium, and large.
```
-- data
   --JHU_low
     -- train_data
      -- ground-truth
      -- images
     -- validation_data
      -- ground-truth
      -- images
   --JHU_medium
     -- train_data
      -- ground-truth
      -- images
     -- validation_data
      -- ground-truth
      -- images
   --JHU_high
     -- train_data
      -- ground-truth
      -- images
     -- validation_data
      -- ground-truth
      -- images
```
2. Data preprocess

   JHU data is segregated into low, medium, and high using the following code file

```
python initial_labeling.py --dataset <dataset name: jhu> --input-dataset-path <original data directory> --output-dataset-path <new_data>/<JHU_medium><JHU_high><JHU_low> 
```

3. Training

```
python train.py --dataset <dataset name: jhu> --data-dir <path to dataset> --device <gpu device id>
```

4. Test(not needed)

```
python test_original.py --model-path <path of the model to be evaluated> --data-path <directory for the dataset> --dataset <dataset name: qnrf, sha, shb or nwpu>
```
**Stage 2:**
1. Data directory structure
  Data to train the classifier
```
-- new_data
   --train
     -- M1
     -- M2
     -- M3
   --validation
     -- M1
     -- M2
     -- M3
```
2. Data preprocess

   JHU data is segregated into M1, M2, and M3 with the help of models pre-trained in the previous stage using the following code file

```
python dm_count_classifier_2nd_labels.py --dataset <dataset name: jhu> --input-dataset-path <original data directory> --output-dataset-path <M1><M2><M3> 
```

3. Training
  classifier Resnet-18 trained using the following file
```
python crowd_classification_2nd classifier.py --dataset <dataset name: jhu> --data-dir <path to dataset> --device <gpu device id>
```

**Stage 3:**
1. Data directory structure
   
```
-- data
   --test_data
      -- ground-truth
      -- images
```

2. Testing

  Here we use all models pre-trained in previous stages to get the final prediction on test images.

```
python crowd_classification_2nd classifier.py --dataset <dataset name: jhu> --data-dir <path to dataset> --device <gpu device id>
```
