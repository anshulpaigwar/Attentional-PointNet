
"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



from __future__ import print_function
from __future__ import division

import argparse
import os
import shutil
import time
import torch

from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Ros Includes
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker,MarkerArray

import sensor_msgs.point_cloud2 as pcl2
import pcl
from pcl_helper import *


import shapely.geometry
import shapely.affinity




'''
Save the model for later
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count





def accuracy(output, target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.view(target.size(0)).long()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res





def binary_accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""

    pred = output >= 0.8
    truth = target >= 0.8
    acc = float(pred.eq(truth).sum()) / float(target.numel())
    return acc




def ros_pub_cloud(points,topic,color,pcl_pub):

    # points = points.cpu().detach().numpy()
    cloud_msg = xyzrgb_array_to_pointcloud2(points,color,stamp =rospy.Time.now(), frame_id = "/zoe/base_link" )
    # rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)
    # rospy.sleep(0.1)


marker_pub = rospy.Publisher("/marker_topic", Marker, queue_size=10)
def ros_pub_marker(loc,color = "green"):
    val = 0.1
    if (color == "green"):
        val = 0.9

    # print(loc[3])
    quat = quaternion_from_euler(0, 0, loc[3])
    marker = Marker()
    marker.header.frame_id = "/zoe/base_link"
    # marker.header.frame_id = "/zoe/base_link"
    marker.type = marker.CUBE
    marker.action = marker.ADD

    marker.scale.x = 2.5
    marker.scale.y = 5
    marker.scale.z = 2
    if (len(loc) > 4):
        marker.scale.x = loc[5]
        marker.scale.y = loc[6]
        marker.scale.z = loc[4]

    marker.color.a = 0.4
    marker.color.r = 1 - val
    marker.color.g = val
    marker.color.b = 0.2
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.pose.position.x = loc[0]
    marker.pose.position.y = loc[1]
    marker.pose.position.z = loc[2]
    # marker.lifetime = rospy.Duration(0.3)
    marker_pub.publish(marker)
    # rospy.sleep(0.008)
    # # while (True):
    # for i in range(2):
    #     marker_pub.publish(marker)
    # # rospy.sleep(0.1)







def rosMarker(loc,idx,color = "green",dur=0.2):
    val = 0.1
    if (color == "green"):
        val = 0.9

    # print(loc[3])
    quat = quaternion_from_euler(0, 0, loc[3])
    marker = Marker()
    # marker.header.frame_id = "map"
    marker.header.frame_id = "/zoe/base_link"
    marker.type = marker.CUBE
    marker.action = marker.ADD

    marker.scale.x = 2.5
    marker.scale.y = 5
    marker.scale.z = 2
    if (len(loc) > 4):
        marker.scale.x = loc[5]
        marker.scale.y = loc[6]
        marker.scale.z = loc[4]

    marker.color.a = 0.4
    marker.color.r = 1 - val
    marker.color.g = val
    marker.color.b = 0.2
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.pose.position.x = loc[0]
    marker.pose.position.y = loc[1]
    marker.pose.position.z = loc[2]
    marker.lifetime = rospy.Duration(dur)
    marker.id = idx
    return marker
    # rospy.sleep(0.008)
    # # while (True):
    # for i in range(2):
    #     marker_pub.publish(marker)
    # # rospy.sleep(0.1)





class RotatedRect:
    def __init__(self, cx, cy, w, h, angle):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h
        self.angle = angle

    def get_contour(self):
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w/2.0, -h/2.0, w/2.0, h/2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.cx, self.cy)

    def intersection(self, other):
        return self.get_contour().intersection(other.get_contour())



def percent_overlap(loc_gt,loc_D,  W = 5, H = 2.5):

    r1 = RotatedRect(loc_gt[0],loc_gt[1],loc_gt[6],loc_gt[5],loc_gt[3]) # x,y,l,w,theta
    r2 = RotatedRect(loc_D[0],loc_D[1],loc_D[6],loc_D[5],loc_D[3]) # x,y,l,w,theta
    area_intersect = r1.intersection(r2).area
    area_union = (loc_gt[6]*loc_gt[5] + loc_D[6]* loc_D[5])- area_intersect
    IoU = area_intersect/area_union
    overlap = area_intersect/(loc_gt[6]*loc_gt[5])
    return IoU, overlap






def bbox_iou(box1, box2, x1y1x2y2=False): # box1 = x,y,z,theta,h,w,l
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[6]/2.0, box2[0]-box2[6]/2.0)
        Mx = max(box1[0]+box1[6]/2.0, box2[0]+box2[6]/2.0)
        my = min(box1[1]-box1[5]/2.0, box2[1]-box2[5]/2.0)
        My = max(box1[1]+box1[5]/2.0, box2[1]+box2[5]/2.0)
        w1 = box1[6]
        h1 = box1[5]
        w2 = box2[6]
        h2 = box2[5]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea







def nms(boxes, nms_thresh):
    if len(boxes) == 0:
        return boxes

    det_confs = torch.zeros(len(boxes))
    for i in range(len(boxes)):
        det_confs[i] = 1-boxes[i][4]

    _,sortIds = torch.sort(det_confs)
    out_boxes = []
    for i in range(len(boxes)):
        box_i = boxes[sortIds[i]]
        if box_i[4] > 0:
            out_boxes.append(box_i)
            for j in range(i+1, len(boxes)):
                box_j = boxes[sortIds[j]]
                if bbox_iou(box_i, box_j, x1y1x2y2=False) > nms_thresh:
                    #print(box_i, box_j, bbox_iou(box_i, box_j, x1y1x2y2=False))
                    box_j[4] = 0
    return out_boxes

























def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def adjust_learning_rate2(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = float(args.lr) / 4.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def visualize(data, fig_num, title):
    input_tensor = data.cpu()
    input_tensor = torch.squeeze(input_tensor)
    in_grid = input_tensor.detach().numpy()

    fig=plt.figure(num = fig_num)
    plt.imshow(in_grid, cmap='gray', interpolation='none')
    plt.title(title)
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    plt.show(block=False)
    # time.sleep(4)
    # plt.close()



def visualize_stn(data, title):
    input_tensor = data.cpu()
    input_tensor = torch.squeeze(input_tensor)
    input_tensor = input_tensor.detach().numpy()
    N = len(input_tensor)

    fig=plt.figure(num = 2)

    columns = N
    rows = 1
    for i in range(1, columns*rows +1):
        img = input_tensor[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray', interpolation='none')

    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    # figManager.window.state('zoomed')
    plt.show(block=False)
    # time.sleep(4)
    # plt.close()
