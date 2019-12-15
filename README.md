# SYDE_671 Final project

## Our project is focused on the following ICCV 2019 paper:
Video Object Segmentation using Space-Time Memory Networks
Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
[[paper]](https://arxiv.org/abs/1904.00607)

[![Video Object Segmentation using Space-Time Memory Networks (ICCV 2019)](https://img.youtube.com/vi/vVZiBEDmgIU/0.jpg)](https://www.youtube.com/watch?v=vVZiBEDmgIU "Video Object Segmentation using Space-Time Memory Networks (ICCV 2019)")

### - Code
- model.py [contains the original STM model implementation + The NEW DECODER ARCHITECTURE]
- helpers.py [helper functions for loading model parameters and segmentation scoring]
- video.py [functions to perform forward pass through video sequence]
- dataset.py [classes to construct DAVIS 2016 dataloaders]
- track.py [class to track results in the form of log file and display them]
- train.py [functions to train network in batches]
- train_DAVIS_2016.py [example script to train or validate on the DAVIS 2016 dataset.]

### - Requirements
- python 3+
- pytorch 1.3.1+
- numpy
- tqdm
- pillow
- scikit-image

### - How to Use
- Download the DAVIS 2016 dataset from https://davischallenge.org/davis2016/code.html
- look in train_DAVIS_2016.py for example.
- Setup the right paths to the data.
- Change settings as described. 
- Run on single GPU. [tested on 1050 ti (4GB)]
