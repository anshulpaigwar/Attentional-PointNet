from __future__ import print_function, division
import os
import sys
import numpy as np
import h5py
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from numpy.linalg import inv
from random import uniform
import random
import math
import cv2
# Ros Includes
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker,MarkerArray



import sensor_msgs.point_cloud2 as pcl2
import pcl
from pcl_helper import *
import ipdb as pdb




rospy.init_node('pcl2_pub_example', anonymous=True)
pcl_pub = rospy.Publisher("/all_points", PointCloud2, queue_size=10)
marker_pub = rospy.Publisher("/marker_topic", Marker, queue_size=10)
# rospy.sleep(0.1)








def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith('.' + extension))


def ros_pub(points,topic,color):
    # points = points.cpu().detach().numpy()
    cloud_msg = xyzrgb_array_to_pointcloud2(points,color,stamp =rospy.Time.now(), frame_id = "/zoe/base_link" )
    rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)
    rospy.sleep(0.1)



def ros_pub_marker(loc):
    # print(loc[3])
    quat = quaternion_from_euler(0, 0, loc[3])
    marker = Marker()
    marker.header.frame_id = "/zoe/base_link"
    marker.type = marker.CUBE
    marker.action = marker.ADD
    marker.scale.x = 2.5
    marker.scale.y = 5
    marker.scale.z = 2
    marker.color.a = 0.4
    marker.color.r = 0.9
    marker.color.g = 0.1
    marker.color.b = 0.2
    marker.pose.orientation.x = quat[0]
    marker.pose.orientation.y = quat[1]
    marker.pose.orientation.z = quat[2]
    marker.pose.orientation.w = quat[3]
    marker.pose.position.x = loc[0]
    marker.pose.position.y = loc[1]
    marker.pose.position.z = loc[2]
    # marker.lifetime = rospy.Duration(1)
    rospy.sleep(0.2)
    # while (True):
    for i in range(5):
        marker_pub.publish(marker)
    # rospy.sleep(0.1)


def rosMarker(loc,idx,color = "green"):
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
    marker.id = idx
    return marker




# # xmin,ymin,zmin,smin,xmax,ymax,zmax,smax
# regions =  [(0, -2, -3, 0, 10, 8, 0,0),
#             (0, -8, -3, 0, 10, 2, 0,0),
#             (10, -2, -3, 0, 20, 8, 0,0),
#             (10, -8, -3, 0, 20, 2, 0,0)]






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

    # for i in range(rows):
    #     for j in range(i+1):
    #         xmin = 0 + i * sz
    #         ymin = -sz/2 + j * sz
    #         coord = (xmin-ovr, ymin-ovr, -3, 0, xmin + sz +ovr, ymin + sz +ovr, 0, 0)
    #         regions.append(coord)

    # for i in range(rows):
    #     for j in range(i):
    #         xmin = 0 + i * sz
    #         ymin = -sz/2 - j * sz
    #         coord = (xmin-ovr, ymin -ovr - sz, -3, 0, xmin + sz +ovr, ymin +ovr, 0, 0)
    #         regions.append(coord)

    return regions


regions = get_regions(sz=10, ovr = 1, rows = 4)
print(regions)


pdb.set_trace()








def lidar_to_img(points, img_size):
    # pdb.set_trace()
    lidar_data = np.array(points[:, :2]) # neglecting the z co-ordinate
    lidar_data *= 9.9999 # multiplying by the resolution
    lidar_data -= (0.5 * img_size, 0.5 * img_size) # recentering the lidar data so that the center is in a corner
    lidar_data = np.fabs(lidar_data)
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img = np.zeros((img_size, img_size))
    lidar_img[tuple(lidar_data.T)] = 255
    return lidar_img



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
    return lidar_img






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


    return crop.cpu().numpy()
























# or (str(labels[i][0]) == "Van")



def check_for_car(labels, calib,r ):
    m = 0 #margin
    num_objects = 3
    car_list = []
    car_count = 0

    t = np.eye(4,4)
    t[0,3] -= (r[0] + r[4])/2.0
    t[1,3] -= (r[1] + r[5])/2.0
    t[2,3] += 1.8

    for i in range(len(labels)):
        if ((str(labels[i][0]) == "Car") ):
            loc = np.array([labels[i][11],labels[i][12],labels[i][13],1]).T
            loc = np.dot(calib,loc)
            if((r[0]+m < loc[0]< r[4]-m) and (r[1]+m < loc[1]< r[5]-m)): # xmin, xmax, ymin, ymax
                loc = np.dot(t,loc)
                theta = math.degrees(-labels[i][14])
                if(theta < 0):
                    theta += 360
                th_bin = int(theta//10)
                car_list.append([loc[0] , loc[1], loc[2]+0.5, -labels[i][14],th_bin, labels[i][8],labels[i][9], labels[i][10], 1])
                # x,y,z,theta, theta bin, h,w,l, class
    return car_list






def kitti_custom_data_generate(data_dir, data_out_dir,frames, num_points = 4096, seq_len= 3):
    crop_num = 0
    velodyne_dir = data_dir + "velodyne/"
    label_dir = data_dir + 'label_2/'
    calib_dir = data_dir + 'calib/'
    last_crop_has_car = False
    for frame in frames:

        points_path = os.path.join(velodyne_dir, "%06d.bin" % frame)
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
        points = points[:, :3]  # exclude luminance
        points = np.array([p for p in points if p[0] > abs(p[1])])

        labels = np.genfromtxt(label_dir + "%06d.txt" % frame, delimiter=' ', dtype=None)
        labels = np.atleast_1d(labels)
        calib = np.genfromtxt(calib_dir + "%06d.txt" % frame, delimiter=' ', skip_header = 5)
        calib = calib[0,1:].reshape(-1, 4)
        calib = inv(np.vstack((calib,[0,0,0,1]))) # needed to stack last row to calib so that to make it square matrix
        for r in regions:
            car_list = check_for_car(labels,calib,r)
            num_cars = len(car_list)
            if(num_cars <= seq_len):
                if( (0 < num_cars)|last_crop_has_car):
                    num_rand_patch = seq_len - num_cars
                    for p in range(num_rand_patch):
                        car_list.append([uniform(-5,5), uniform(-5,5),uniform(0.5,1.5),uniform(0,3.14),0,2,2.5,5,0])

                    crop = load_velodyne_points_torch(points,r, num_points) #COMBAK: very ineffeicient bcoz loding the pointcloud 4 times
                    # crop_img = lidar_to_img(crop, img_size=120)
                    crop_img = lidar_to_heightmap(crop, img_size=120)
                    crop_path = data_out_dir + "velodyne_crop/" + "%06d" % crop_num
                    crop_label_path = data_out_dir + "labels/" + "%06d.txt" % crop_num
                    crop_img_path = data_out_dir + "heightmap_crop/" + "%06d.png" % crop_num

                    cv2.imwrite(crop_img_path, crop_img)


                    np.save(crop_path,crop) # CHANGED
                    np.savetxt(crop_label_path, car_list, delimiter=' ')
                    # print(crop_num)
                    crop_num +=1
            if(num_cars > 0):
                last_crop_has_car = True
            else:
                last_crop_has_car = False








markerArray_pub = rospy.Publisher("/markerArray_topic", MarkerArray, queue_size=10)

def kitti_custom_data_visualise(data_dir):
    markerArray = MarkerArray()
    files = os.listdir(data_dir + "velodyne_crop/")
    print(len(files))
    for crop_num in range(len(files)):
        crop_path = data_dir + "velodyne_crop/"+ "%06d.npy" % crop_num
        points = np.load(crop_path)
        N = points.shape[0] # Num of points in PointCloud
        color = np.array([1,1,1])
        color = np.vstack([color]*N)
        print(N)
        ros_pub(points, "/all_points", color)


        labels_path = data_dir + "labels/" + "%06d.txt" % crop_num
        labels = np.loadtxt(labels_path, delimiter=' ',ndmin=2)
        for x in range(len(labels)):
            loc = labels[x]
            if(loc[8] == 1):
                marker = rosMarker(loc,x, "green")
            else:
                loc[:4] = np.array([0, 8, 1.5, 3.14/2.0])
                marker = rosMarker(loc,x, "red")

            markerArray.markers.append(marker)
            # ros_pub_marker(labels[x])
            # pdb.set_trace()
        markerArray_pub.publish(markerArray)
        pdb.set_trace()




if __name__ == '__main__':
    data_dir = "/home/anshul/inria_thesis/datasets/kitti/data_object_velodyne/training/"
    train_out_dir = "/home/anshul/iros_2019/attentional_pointnet/my_dataset/corrected_data/training/"
    valid_out_dir = "/home/anshul/iros_2019/attentional_pointnet/my_dataset/corrected_data/validation/"
    total_frames = range(7480)
    random.seed(720)
    random.shuffle(total_frames)
    print(total_frames[:10])
    split = int(np.floor(0.7 * 7480))
    training_frames = total_frames[:split]
    validation_frames = total_frames[split:]

    kitti_custom_data_generate(data_dir,train_out_dir,training_frames,4096)
    kitti_custom_data_generate(data_dir,valid_out_dir,validation_frames,4096)
    kitti_custom_data_visualise(valid_out_dir)















#
#
# def check_for_objects(labels, calib,r ):
#     m = 1 #margin
#     car_count = 0
#     pedestrian_count = 0
#     cyclist_count = 0
#     car_list = []
#     for i in range(len(labels)):
#         if ((str(labels[i][0]) == "Car") or (str(labels[i][0]) == "Van")):
#             car_count +=1
#         elif (str(labels[i][0]) == "pedestrian")
#
#
#
#             loc = np.array([labels[i][11],labels[i][12],labels[i][13],1]).T
#             loc = np.dot(calib,loc)
#             if((r[0]+m < loc[0]< r[4]-m) and (r[1]+m < loc[1]< r[5]-m)): # xmin, xmax, ymin, ymax
#                 car_list.append([loc[0] , loc[1], loc[2]+0.5])
#     return np.asarray(car_list)
#
#
#
# def kitti_mix_data_generate(self, data_dir, data_out_dir):
#     crop_num = 0
#     velodyne_dir = data_dir + "velodyne/"
#     label_dir = data_dir + 'label_2/'
#     calib_dir = data_dir + 'calib/'
#     for frame in range(80,200):
#         labels = np.genfromtxt(label_dir + "%06d.txt" % frame, delimiter=' ', dtype=None)
#         labels = np.atleast_1d(labels)
#         calib = np.genfromtxt(calib_dir + "%06d.txt" % frame, delimiter=' ',skip_header = 5)
#         calib = calib[0,1:].reshape(-1, 4)
#         calib = inv(np.vstack((calib,[0,0,0,1])))
#         for region in range(4):
#             crop = load_velodyne_points(velodyne_dir,frame,region)
#             object_list = check_for_objects(labels,calib,regions[region]) # atleast two car or a pedestrian or a cyclyist
#
#
#
#
#
#
#             car_list = check_for_car(labels,calib,regions[region])
#             if(car_list.size != 0):
#                 crop_path = data_out_dir + "velodyne_crop/" + "%06d" % crop_num
#                 crop_label_path = data_out_dir + "labels/" + "%06d.txt" % crop_num
#                 np.save(crop_path,crop)
#                 np.savetxt(crop_label_path, car_list, delimiter=' ')
#                 # print(crop_num)
#                 crop_num +=1
#
#



# def resize_recenter_points(crop, r, num_points):
#     t = np.eye(4)
#     t[0,3] -= (r[0] + r[4])/2.0
#     t[1,3] -= (r[1] + r[5])/2.0
#     t[2,3] += 1.8
#     crop = np.hstack((crop,np.ones((len(crop),1))))
#     crop =  np.dot(crop, t.T)
#     crop = crop[:,:3]
#










#
# def kitti_custom_data_generate(data_dir, data_out_dir, num_points = 4096):
#     crop_num = 0
#     velodyne_dir = data_dir + "velodyne/"
#     label_dir = data_dir + 'label_2/'
#     calib_dir = data_dir + 'calib/'
#     for frame in range(0,7480):
#         labels = np.genfromtxt(label_dir + "%06d.txt" % frame, delimiter=' ', dtype=None)
#         labels = np.atleast_1d(labels)
#         calib = np.genfromtxt(calib_dir + "%06d.txt" % frame, delimiter=' ', skip_header = 5)
#         calib = calib[0,1:].reshape(-1, 4)
#         calib = inv(np.vstack((calib,[0,0,0,1]))) # needed to stack last row to calib so that to make it square matrix
#         last_crop_has_car = True
#         for region in range(4):
#             r = regions[region]
#             car_list = check_for_car(labels,calib,r)
#             num_cars = len(car_list)
#             if(num_cars < 4):
#                 if( (0 < num_cars)):
#                     num_rand_patch = 3 - num_cars
#                     for p in range(num_rand_patch):
#                         car_list.append([uniform(-5,5), uniform(-5,5),uniform(0.5,1.5),uniform(-3.14,3.14), 0])
#
#                     crop = load_velodyne_points_ransac(velodyne_dir,frame,r, num_points) #COMBAK: very ineffeicient bcoz loding the pointcloud 4 times
#                     crop_path = data_out_dir + "velodyne_crop/" + "%06d" % crop_num
#                     crop_label_path = data_out_dir + "labels/" + "%06d.txt" % crop_num
#
#
#                     # recenter resize the pointcloud:
#
#                     # crop = resize_recenter_points(crop, r, num_points)
#
#                     np.save(crop_path,crop)
#                     np.savetxt(crop_label_path, car_list, delimiter=' ')
#                     # print(crop_num)
#                     crop_num +=1
#
#             if(num_cars > 0):
#                 last_crop_has_car = True
#             else:
#                 last_crop_has_car = False
#










# def load_velodyne_points_ransac(velodyne_dir, frame, r, num_points):
#     points_path = os.path.join(velodyne_dir, "%06d.bin" % frame)
#     points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 4)
#     points = points[:, :3]  # exclude luminance
#
#     cloud = pcl.PointCloud()
#     cloud.from_array(points)
#     clipper = cloud.make_cropbox()
#     clipper.set_MinMax(*r)
#     out_cloud = clipper.filter()
#
#     # np_cloud = np.asarray(out_cloud)
#     # N = np_cloud.shape[0] # Num of points in PointCloud
#     # color = np.array([1,1,1])
#     # color = np.vstack([color]*N)
#     # print(N)
#     # ros_pub(np_cloud, "/all_points", color)
#     #
#     # pdb.set_trace()
#
#
# ################ Voxel Grid Filter ################
#
#     if(out_cloud.size > 15000):
#
#         # resize PointCloud
#         leaf_size = 0.05
#         vox = out_cloud.make_voxel_grid_filter()
#         vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
#         out_cloud = vox.filter()
#         # print(out_cloud.size)
#
#
#     seg = out_cloud.make_segmenter()
#     seg.set_model_type(pcl.SACMODEL_PLANE)
#     seg.set_method_type(pcl.SAC_RANSAC)
#     seg.set_distance_threshold(0.2)
#     indices, model = seg.segment()
#     out_cloud = out_cloud.extract(indices, negative = True)
#
#     out_cloud = np.asarray(out_cloud)
#     if(out_cloud.shape[0] > num_points):
#         choice = np.random.choice(len(out_cloud), num_points, replace=False)
#         out_cloud = out_cloud[choice, :]
#     elif(out_cloud.shape[0] <= num_points/2):
#         choice = np.random.choice(len(out_cloud), num_points - len(out_cloud), replace=True)
#         extra_points = out_cloud[choice, :]
#         out_cloud = np.vstack((out_cloud,extra_points))
#     else:
#         choice = np.random.choice(len(out_cloud), num_points - len(out_cloud), replace=False)
#         extra_points = out_cloud[choice, :]
#         out_cloud = np.vstack((out_cloud,extra_points))
#
#
#     x_shift = (r[0] + r[4])/2.0
#     y_shift = (r[1] + r[5])/2.0
#     z_shift = -1.8
#     translation = np.array([x_shift,y_shift,z_shift])
#     out_cloud = np.subtract(out_cloud,translation)
#
#
#
#     # N = out_cloud.shape[0] # Num of points in PointCloud
#     # color = np.array([1,1,1])
#     # color = np.vstack([color]*N)
#     # print(N)
#     # ros_pub(out_cloud, "/all_points", color)
#     #
#     # pdb.set_trace()
#
#
#     return out_cloud
#
#
#
#
