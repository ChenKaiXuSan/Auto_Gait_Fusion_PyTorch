<div align="center">    
 
# Gait Cycle based Action Recognition for Adult Spinal Deformity Classification
  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
<!-- ![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push) -->

<!--  
Conference   
-->
</div>
 
``` mermaid
graph LR
A[video] --> B(location detector)
B --> C[Human area]
C --> D(gait definition)

D --> stance[stance phase]
D --> swing[swing phase]

stance --> auto[auto fuse]
swing --> auto[auto fuse]

auto --> cnn[Classification]
cnn --> result[predict result]
```

## Description

The main contribute of this project is:

1. use pose estimation to define the gait cycle for different person.
2. split the gait cycle into several parts.
3. defined a method to auto find the positive gait phase.
4. use the trained model to classify the disease.

The classification label is:

- ASD
- DHS
- HipOA + LCS
- Normal

The reason why I combine the HipOA with LCS has two point,

1. data is not enough.
2. the HipOA and LCS are similar in the medical expression.

## Gait Defined Model

Define the **One Gait Cycle** is a key point in this study.
A simple but effective way is to use the **pose estimation**.
For example, use one certain keypoint (left foot, etc.) to define the gait cycle.

![define one gait cycle](images/gait_cycle.png)

To estimate the keypoint, we try to use some different method to predict the keypoint.

1. YOLOv8 pose estimation
2. mediapipe pose estimation
3. openpose

But, there also have some problem in this method.

## Auto fuse the gait cycle

``` mermaid
graph LR
A[defined gait cycle] --> swing(swing phase frame)
swing --> st_cnn[stance model]

st_cnn --> trained_st[trained stance model]
st_cnn --> sw_pred[predict]
sw_pred <--> label

A --> stance(stance phase frame)
stance --> sw_cnn[swing model]
sw_cnn --> st_pred[predict]
st_pred <--> label

sw_cnn --> trained_sw[trained swing model]

```

We first use the stance phase frames to train the stance model, and use the swing phase frames to train the swing model.
Then, we use the trained model to predict the gait cycle.

<!-- 
## Citation

```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
``` -->
