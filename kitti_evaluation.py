#!/usr/bin/env python
"""
@Author: Anshul Paigwar
@email: p.anshul6@gmail.com

For more information on python-pcl check following links:

Git Hub repository:
https://github.com/strawlab/python-pcl
Check the examples and tests folder for sample coordinates

API documentation:
http://nlesc.github.io/python-pcl/
documentation is incomplete there are more available funtions

Udacity Nanodegree perception exercises for practice
https://github.com/udacity/RoboND-Perception-Exercises

check the documentation for pcl_helper.py

"""


from __future__ import print_function

# Ros imports:
import rospy
import tf
import math
import sys

from sensor_msgs.msg import PointCloud2
import std_msgs.msg
import sensor_msgs.point_cloud2 as pcl2
from visualization_msgs.msg import Marker,MarkerArray

import ipdb as pdb
import argparse
import random
import time
import os
import numpy as np
from numpy.linalg import inv
import pcl
from tools.pcl_helper import *


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from modules import A3D_Loss_New,A3D_Loss
from model import AttentionalPointnet
from tools.utils import ros_pub_marker, ros_pub_cloud, AverageMeter, rosMarker,percent_overlap, nms
import multiprocessing as mp
from numba import njit, jit

markerArray_pub = rospy.Publisher("/markerArray_topic", MarkerArray, queue_size=10)
pcl_pub = rospy.Publisher("/all_points", PointCloud2, queue_size=10)

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
args = parser.parse_args()

args.resume = "/home/anshul/iros_2019/attentional_pointnet/my_codes/atten_img_pointnet/checkpoint.pth.tar"



seq_len = 3
model = AttentionalPointnet(N = 4096)
if use_cuda:
    model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))






def get_regions(sz=10, ovr = 1,rows = 3 ): # Sz = size Ovr = overlap

    """
    Creates a list of co-ordinates of the cropped regions
    coordinates = (xmin, ymin, zmin, smin, xmax, ymax, zmax, smax)

    Args
    ----
    - sz: size of the cropped region
    - ovr: overlap size

    Returns
    -------
    - regions: a 4D Tensor of shape (num of regions, 8)
    """
    regions =[]
    for i in range(rows):
        for j in range(i+1):
            xmin = 0 + i * sz
            ymin = 0 + j * sz
            coord = (xmin-ovr, ymin-ovr, -3, 0, xmin + sz +ovr, ymin + sz +ovr, 0, 0)
            regions.append(coord)

    for i in range(rows):
        for j in range(i+1):
            xmin = 0 + i * sz
            ymin = 0 - j * sz
            coord = (xmin-ovr, ymin -ovr - sz, -3, 0, xmin + sz +ovr, ymin +ovr, 0, 0)
            regions.append(coord)
    return regions


regions = get_regions(sz=10, ovr = 1, rows = 4)
# print(regions)
















def check_for_car(labels, calib ):
    car_list = []
    for i in range(len(labels)):
        if ((str(labels[i][0]) == "Car") and  (labels[i][2] ==2)):
            loc = np.array([labels[i][11],labels[i][12],labels[i][13],1]).T
            loc = np.dot(calib,loc)
            car_list.append([loc[0] , loc[1], loc[2]+0.5, -labels[i][14],labels[i][8],labels[i][9], labels[i][10]])
    return car_list





def lidar_to_img(points, img_size):
    # pdb.set_trace()
    lidar_data = np.array(points[:, :2])
    lidar_data *= 9.9999
    lidar_data -= (0.5 * img_size, 0.5 * img_size)
    lidar_data = np.fabs(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img = np.zeros((img_size, img_size))
    lidar_img[tuple(lidar_data.T)] = 255
    return torch.tensor(lidar_img).cuda()


# def lidar_to_img(points, img_size):
#     # pdb.set_trace()
#     lidar_data = points[:, :2]
#     lidar_data *= 9.9999
#     lidar_data -= torch.tensor((0.5 * img_size, 0.5 * img_size)).cuda()
#     lidar_data = torch.abs(lidar_data)
#     lidar_data = torch.floor(lidar_data).long()
#     lidar_data = lidar_data.view(-1, 2)
#     lidar_img = torch.zeros((img_size, img_size)).cuda()
#     lidar_img[lidar_data.permute(1,0)] = 255
#     return lidar_img



def lidar_to_heightmap(points, img_size):
    # pdb.set_trace()
    lidar_data = np.array(points[:, :2])
    height_data = np.array(points[:,2])
    height_data *= 255/2
    height_data[height_data < 0] = 0
    height_data[height_data > 255] = 255
    height_data = np.fabs(height_data)
    height_data = height_data.astype(np.int32)


    lidar_data *= 9.9999
    lidar_data -= (0.5 * img_size, 0.5 * img_size)
    lidar_data = np.fabs(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img = np.zeros((img_size, img_size))
    lidar_img[tuple(lidar_data.T)] = height_data # TODO: sort the point wrt height first lex sort
    return torch.tensor(lidar_img).cuda()





# def lidar_to_heightmap(points, img_size = 120):
#     # pdb.set_trace()
#     lidar_data = points[:, :2]
#     height_data = points[:, 2]
#     height_data *= 255/2
#     height_data[height_data < 0] = 0
#     height_data[height_data > 255] = 255
#     height_data = torch.abs(height_data)
#     height_data = torch.floor(height_data)
#
#     lidar_data *= 9.9999
#     lidar_data -= torch.tensor((0.5 * img_size, 0.5 * img_size)).cuda()
#     lidar_data = torch.abs(lidar_data)
#     lidar_data = torch.floor(lidar_data).long()
#     lidar_data = lidar_data.view(-1, 2)
#     lidar_img = torch.zeros((img_size, img_size)).cuda()
#     lidar_img[tuple(lidar_data.permute(1,0))] = height_data# TODO: sort the point wrt height first lex sort
#     return lidar_img










def load_velodyne_points_torch(points, rg, N):

    cloud = pcl.PointCloud()
    cloud.from_array(points)
    clipper = cloud.make_cropbox()
    clipper.set_MinMax(*rg)
    out_cloud = clipper.filter()

    if(out_cloud.size > 15000):
        leaf_size = 0.05
        vox = out_cloud.make_voxel_grid_filter()
        vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
        out_cloud = vox.filter()

    if(out_cloud.size > 0):
        cloud = torch.from_numpy(np.asarray(out_cloud)).float().cuda()

        points_count = cloud.shape[0]
        # pdb.set_trace()
        # print("indices", len(ind))
        if(points_count > 1):
            prob = torch.randperm(points_count)
            if(points_count > N):
                idx = prob[:N]
                crop = cloud[idx]
                # print(len(crop))
            else:
                r = int(N/points_count)
                cloud = cloud.repeat(r+1,1)
                crop = cloud[:N]
                # print(len(crop))

            x_shift = (rg[0] + rg[4])/2.0
            y_shift = (rg[1] + rg[5])/2.0
            z_shift = -1.8
            shift = torch.tensor([x_shift, y_shift, z_shift]).cuda()
            crop = torch.sub(crop, shift)

        else:
            crop = torch.ones(N,3).cuda()
            # print("points count zero")

    else:
        crop = torch.ones(N,3).cuda()
        # print("points count zero")


    return crop










def get_data(data_dir, frame, batch_size, num_points):
    velodyne_dir = data_dir + "velodyne/"
    label_dir = data_dir + 'label_2/'
    calib_dir = data_dir + 'calib/'
    points_path = os.path.join(velodyne_dir, "%06d.bin" % frame)
    points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
    points = points[:, :3]  # exclude luminance
    points = np.array([p for p in points if p[0] > abs(p[1])])

    labels = np.genfromtxt(label_dir + "%06d.txt" % frame, delimiter=' ', dtype=None)
    labels = np.atleast_1d(labels)
    calib = np.genfromtxt(calib_dir + "%06d.txt" % frame, delimiter=' ', skip_header = 5)
    calib = calib[0,1:].reshape(-1, 4)
    calib = inv(np.vstack((calib,[0,0,0,1]))) # needed to stack last row to calib so that to make it square matrix

    car_list = check_for_car(labels,calib)
    data = []
    img_data = []



    # for B in range(batch_size):
    #     r = regions[B]
    for r in regions:
        # r = regions[B]
        # crop_start = time.time()
        cropped_points = load_velodyne_points_torch(points,r, num_points)
        # crop_time_points = time.time()
        # print('velodyne_time:', crop_time_points - crop_start)
        crop_img = lidar_to_heightmap(cropped_points.cpu(), img_size=120)
        # crop_time_img = time.time()
        # print('img_time:', crop_time_img - crop_time_points)

        # crop_img = lidar_to_img(cropped_points.cpu(), img_size=120)
        data.append(cropped_points)
        img_data.append(crop_img)




    extras = batch_size - len(regions)
    if extras:
        data.extend(data[:extras])
        img_data.extend(img_data[:extras])

    # for i in range(batch_size - len(regions)):
    #     cropped_points = torch.ones(num_points,3).cuda()
    #     crop_img = lidar_to_heightmap(cropped_points, img_size=120)
    #     # crop_img = lidar_to_img(cropped_points.cpu(), img_size=120)
    #     data.append(cropped_points)
    #     img_data.append(crop_img)

    data = torch.stack(data)
    img_data = torch.stack(img_data)

    return points, data, img_data, car_list



def eval_detection_frame(car_list, detections):
    num_cars = len(car_list)
    num_detect = len(detections)
    TP_frame = torch.zeros(num_cars)
    CS_frame = torch.zeros(num_detect)
    for i in range(num_cars):
        for j in range(num_detect):
            IoU, overlap = percent_overlap(car_list[i],detections[j])
            if(overlap >= 0.5):
                TP_frame[i] = 1
                CS_frame[j] = (1+math.cos(car_list[i][3] - detections[j][3]))/2.0
    return TP_frame,CS_frame









def kitti_data_evaluation(frames, num_points = 4096, visualize = True):
    TP = torch.zeros(0) # True Positives
    CS = torch.zeros(0) # Cosine Similarity
    batch_size = 32
    model.eval()
    with torch.no_grad():
        for frame in frames:

            markerArray = MarkerArray()
            start = time.time()
            out_cloud, data, img_data, car_list = get_data(data_dir, frame, batch_size, num_points)
            pre_process_time = time.time()
            print('pre_process_time:', pre_process_time - start)


            B = data.shape[0] # Batch size
            data, img_data  = data.float(), img_data.float()

            img_data  = img_data.unsqueeze(1)
            hidden = torch.zeros(1,B,512).cuda() # initialising the hidden variable for GRU
            # optimizer.zero_grad()

            output = model(data, img_data, hidden, seq_len) # (B,4)


            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            hidden = hidden.detach()
            model_time = time.time()
            print('model_time:', model_time - pre_process_time)


            score_seq, loc_seq, box_seq = output

            trans_mat_1 = torch.eye(3).view(1,-1).repeat(seq_len,B,1).cuda()
            trans_mat_1[:,:,[0,3,4,2,5]] = loc_seq[:,:,[0,1,0,2,3]] # c,s,c,tx,ty,tz
            trans_mat_1[:,:,1] = -loc_seq[:,:,1] # -
            trans_mat_1 = trans_mat_1.view(seq_len*B,3,3)




            trans_mat_2 = torch.eye(3).view(1,-1).repeat(seq_len,B,1).cuda()
            trans_mat_2[:,:,[0,3,4,2,5]] = box_seq[:,:,[0,1,0,2,3]] # c,s,c,tx,ty,tz
            trans_mat_2[:,:,1] = -box_seq[:,:,1] # -s
            trans_mat_2 = trans_mat_2.view(seq_len*B,3,3)


            resultant_trans = torch.bmm(trans_mat_1, trans_mat_2)
            resultant_trans = resultant_trans.view(seq_len,B, 9)

            final_trans_params = resultant_trans[:,:,[0,3,2,5]]
            z = (loc_seq[:,:,4] + box_seq[:,:,4]).view(seq_len,B,-1)
            final_trans_params = torch.cat((final_trans_params,z),2)

            loc = final_trans_params[:,:,2:5]
            theta = torch.atan2(final_trans_params[:,:,1], final_trans_params[:,:,0])
            size = box_seq[:,:,5:]

            counter = 0

            detections = []

            # for a in range(batch_size):
            for a in range(len(regions)):

                # color = np.array([1,1,1])
                # color = np.vstack([color]*num_points)
                # ros_pub_cloud(data[a].cpu(), "/all_points", color,pcl_pub)
                # pdb.set_trace()

                rg = regions[a]
                for i in range(seq_len):
                    if((score_seq[i,a] > 0.9)):
                        x_shift = (rg[0] + rg[4])/2.0
                        y_shift = (rg[1] + rg[5])/2.0
                        z_shift = 0
                        shift = torch.tensor([x_shift, y_shift, z_shift]).cuda()
                        locx = torch.add(loc[i,a], shift)
                        # locx = loc[i,a]
                        trans_params = torch.cat((locx, theta[i,a].view(1), size[i,a],score_seq[i,a].view(1)),0)
                        detections.append(trans_params.cpu().numpy())

            post_process_time = time.time()
            print('post_process_time:', post_process_time - model_time)

            detections = nms(detections,nms_thresh=0.2)
            nms_time = time.time()
            print('nms_time:',  nms_time - post_process_time)
            print('total_time:', nms_time - start)


            if visualize:

                out_cloud = np.asarray(out_cloud)
                out_cloud[:,2] += 1.73
                color = np.array([1,1,1])
                color = np.vstack([color]*out_cloud.shape[0])
                ros_pub_cloud(out_cloud, "/all_points", color,pcl_pub)


                # for id, car in enumerate(car_list):
                #     marker = rosMarker(car,id, "red",dur=10)
                #     markerArray.markers.append(marker)
                # markerArray_pub.publish(markerArray)
                # pdb.set_trace()

                for id, car in enumerate(detections):
                    # ros_pub_marker(trans_params, "green")
                    # pdb.set_trace()
                    marker = rosMarker(car, id, "green", dur = 10)
                    markerArray.markers.append(marker)


                markerArray_pub.publish(markerArray)
                pdb.set_trace()



            TP_frame, CS_frame = eval_detection_frame(car_list, detections)
            TP = torch.cat((TP,TP_frame),0)
            CS = torch.cat((CS,CS_frame),0)







    if(TP.nelement() == 0):
        recall = 0
    else:
        recall = TP.mean()

    if(CS.nelement() == 0):
        AOS = 0
    else:
        AOS = CS.mean()
    print("recall: ", recall)
    print("AOS: ", AOS)



if __name__ == '__main__':
    visualize = True
    if visualize:
        rospy.init_node('listener', anonymous=True)
    data_dir = "/home/anshul/inria_thesis/datasets/kitti/data_object_velodyne/training/"
    total_frames = range(7480)
    random.seed(720)
    random.shuffle(total_frames)
    print(total_frames[:10])
    split = int(np.floor(0.7 * 7480))
    validation_frames = total_frames[split:]
    training_frames = total_frames[:split]

    kitti_data_evaluation(validation_frames, 4096, visualize)
