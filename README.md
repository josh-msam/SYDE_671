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

### - Requirements
- python 3+
- pytorch 1.0.1.post2
- numpy

### - How to Use
#### Download weights
##### Place it the same folder with demo scripts
```
wget -O STM_weights.pth "https://www.dropbox.com/s/mtfxdr93xc3q55i/STM_weights.pth?dl=1"
```

#### Run
##### DAVIS-2016 validation set (Single-object)
``` 
python eval_DAVIS.py -g '1' -s val -y 16 -D [path/to/DAVIS]
```
##### DAVIS-2017 validation set (Multi-object)
``` 
python eval_DAVIS.py -g '1' -s val -y 17 -D [path/to/DAVIS]
```


### - Reference 
If you find our paper and repo useful, please cite our paper. Thanks!
``` 
Video Object Segmentation using Space-Time Memory Networks
Seoung Wug Oh, Joon-Young Lee, Ning Xu, Seon Joo Kim
ICCV 2019
```

### - Related Project
``` 
Fast Video Object Segmentation by Reference-Guided Mask Propagation
Seoung Wug Oh, Joon-Young Lee, Kalyan Sunkavalli, Seon Joo Kim
CVPR 2018
```
[[paper]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Oh_Fast_Video_Object_CVPR_2018_paper.pdf)
[[github]](https://github.com/seoungwugoh/RGMP)



### - Terms of Use
This software is for non-commercial use only.
The source code is released under the Attribution-NonCommercial-ShareAlike (CC BY-NC-SA) Licence
(see [this](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) for details)

