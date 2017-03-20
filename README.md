
## Machine Learning Engineer Nanodegree
### Capstone Project : DeepTesla

This project is based on Course [MIT 6.S094: Deep Learning for Self-Driving Cars](http://selfdrivingcars.mit.edu/deeptesl/) and published on this [Github](https://github.com/nd009/capstone) 

### Problem Statement

The goal is to predict the steering wheel angel from Tesla dataset based on the video of the forward roadway.

### Requirements
python 3 + Keras 2.0.1 + Tensorflow 1.0.1 + Jupyter Notebook + cv2

These models are trained by GPU Intensive workloads with 61G memory and 12G GPU memory on [floydhub.com](https://www.floydhub.com).

The refined modle is trained ~4 hours.

### Datasets and Inputs

Databases with real-traffic video data captured and extracted 10 video clips of highway driving from Tesla:

- The wheel value was extracted from the in-vehicle CAN

- A window from each video frame is cropped/extracted and provide a CSV linking the window to a wheel value.

A snapshot of video frame:
<img src="./images/img/frame_1173.jpg" width = "320" height = "180" align=center />
    
The CSV data format:


|  ts_micro         | frame_index | wheel |
|:-----------------:|:-----------:|:-----:|
|  1464305394391807 | 0           | -0.5  |
| 1464305394425141  | 1           | -0.5  | 
| 1464305394458474  | 2           | -0.5  |


in which, `ts_micro` is time stamp，`frame_index` denotes frame number，`wheel` is steering wheel angle(Based on horizontal, + is clockwise, - is anticlockwise)


![](./images/img/gif_tesla_vgg.gif)
