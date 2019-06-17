## Attentional PointNet for 3D-Object Detection in Point Clouds
Authors: Anshul Paigwar, Ozgur Erkent, Christian Wolf, Christian Laugier

<img src="https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/doc/teaser_final.png" alt="drawing" width="400"/>

## Introduction
This repository is a code release for our CVPR 2019 workshop [paper](https://hal.inria.fr/hal-02156555) accepted in [Workshop on Autonomous Driving](https://sites.google.com/view/wad2019/overview).
In this work, we study 3D object detection directly from point clouds obtained
from 3D LiDARS.

## Abstract
Accurate detection of objects in 3D point clouds is a central
problem for autonomous navigation. Most existing methods use techniques of hand-crafted features representation or multi-sensor approach prone to sensor failure.
Approaches like PointNet that directly operate on a sparse point
data have shown good accuracy in the classification of single 3D objects.
However, LiDAR sensors on Autonomous Vehicles generate a large scale point cloud.
Real-time object detection in such a cluttered environment still remains
a challenge. In this study, we propose Attentional PointNet,
which is a novel end-to-end trainable deep architecture
for object detection in point clouds. We extend the theory
of visual attention mechanisms to 3D point clouds and introduce
a new recurrent 3D Localization Network module.
Rather than processing the whole point cloud, the network
learns where to look (finding regions of interest), which significantly reduces
the number of points to be processed and inference time.
Evaluation on KITTI car detection benchmark shows that our Attentional PointNet achieves comparable results with the state-of-the-art LiDAR-based 3D detection methods in detection and speed.

## Installation

We have tested the algorithm on the system with Ubuntu 16.04, 8 GB RAM
and NVIDIA GTX-1080.
### Dependencies
```
Python 2.7
CUDA 9.1
PyTorch 1.0
scipy
shapely
```
### Visualization
For visualization of the output bounding boxes and easy integration
with our real system we use the Robot Operating System (ROS):
```
ROS
PCL
```
## Data Preparation
* We train the model on augmented KITTI 3D object detection dataset.
* We subdivide the FOV area from each scan into equally
spaced cropped regions of 12m×12m with an overlap of 1m.
* Each cropped region of size 12m×12m is also converted into
a grayscale image of size 120×120 pixels encoding height.

<img src="https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/doc/data_agumentation.png" alt="drawing" width="300"/>


Download the KITTI 3D object detection dataset from [link](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).
  Your data directory should look like:
```
|--data_object_velodyne
    |--training
        |--calib
        |--label_2
        |--velodyne
```
To generate the augmented dataset for the training and validation of Attentional PointNet
 use the code in folder kitti_custom:

 ```
 python kitti_lidarImg_data_generator.py
 ```
It will generate the dataset in the following format:
```
|--attentional_pointnet_data
    |--validation
    |--training
        |--heightmap_crop
            |--0000.png
            |--
        |--labels
            |--0000.txt
            |--
        |--velodyne_crop
            |--0000.npy
            |--
```
Form each cropped region of 12m x 12m we have point cloud data, heightmap and a label file.
 Each label file contains three instances. These instances could be a car or non-car.
 depending upon the number of cars in the scene. Instances are defined as

```
Float x,y,z
Float theta
Int theta_binned
Float H, W, L
Bool category car/ non-car
```
For non-car category, we keep a fixed x,y,z which is outside of 12m x 12m region.

## Usage

To train the model update the path to the data directory in your system and run:
```
python main.py -s
```
It takes around 8 hours for the network to converge and model parameters would be stored
 in model_checkpoint.pth.tar file. You can then use these model parameters for final
 evaluation on KITTI dataset:
```
python kitti_evaluation.py
```
## Results

Attentional PointNet achieves arround 55.6 % of overall Average Precision for car detection
 combined for all levels of difficulties. Some of the results are shown below:

Sequential detection:

<img src="https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/doc/sequence_detection.png" alt="drawing" width="700"/>

Detection on Kitti Dataset:

<img src="https://github.com/anshulpaigwar/Attentional-PointNet/blob/master/doc/kitti_detections_final.png" alt="drawing" width="700"/>

## References

[1] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. Pointnet: Deep learning on
point sets for 3d classification and segmentation. Proc. Computer Vision and Pattern
Recognition (CVPR), IEEE, 2017.
