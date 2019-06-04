from __future__ import print_function, division
import os
import sys
import torch

import cv2

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# Ros Includes
import rospy
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import PointCloud2
import std_msgs.msg
from visualization_msgs.msg import Marker



import sensor_msgs.point_cloud2 as pcl2
import pcl
from pcl_helper import *
import ipdb as pdb








def ros_pub(points,topic,color):
    pcl_pub = rospy.Publisher("/all_points", PointCloud2, queue_size=10)
    points = points.cpu().detach().numpy()
    cloud_msg = xyzrgb_array_to_pointcloud2(points,color,stamp =rospy.Time.now(), frame_id = "map" )
    rospy.loginfo("happily publishing sample pointcloud.. !")
    pcl_pub.publish(cloud_msg)
    rospy.sleep(0.1)



def ros_pub_marker(loc):
    marker_pub = rospy.Publisher("/marker_topic", Marker, queue_size=10)
    # print(loc[3])
    quat = quaternion_from_euler(0, 0, loc[3])
    marker = Marker()
    marker.header.frame_id = "map"
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





class kitti_custom_dataset(Dataset):
    def __init__(self, data_dir, num_points=2048, mode='train'):
        self.data = []
        self.labels = []
        self.img_data = []
        files = os.listdir(data_dir + "velodyne_crop/")
        for crop_num in range(0, len(files)):
            crop_path = data_dir + "velodyne_crop/"+ "%06d.npy" % crop_num
            point_set = np.load(crop_path) #(N,3)
            self.data.append(point_set)

            labels_path = data_dir + "labels/" + "%06d.txt" % crop_num
            pose = np.loadtxt(labels_path, delimiter=' ',ndmin=2) #(3,4)
            self.labels.append(pose)

            img_path = data_dir + "img_crop/" + "%06d.png" % crop_num
            img = cv2.imread(img_path,0)
            self.img_data.append(img)

    def __getitem__(self, index):
        return self.data[index], self.img_data[index], self.labels[index]


    def __len__(self):
        # print(len(self.data))
        return len(self.data)










def get_data_loaders(data_dir):

    datasets = kitti_custom_dataset(data_dir)
    data_len = len(datasets)
    print("Total Data size ", data_len)
    indices = list(range(data_len))

    shuffle = True
    random_seed = 20
    batch_size = 32

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("using cuda")
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 4
        pin_memory = True




    # split the main dataset into three parts test(20%), valid(20%) and train (70%)
    split_train = int(np.floor(0.7 * data_len))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[:split_train]
    valid_idx = indices[split_train:]

    print("Train Data size ",len(train_idx))
    print("Valid Data size ",len(valid_idx))


    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = DataLoader( datasets, batch_size= batch_size, sampler=train_sampler, # BS = 105
        num_workers=num_workers, pin_memory=pin_memory,drop_last=True)
    valid_loader = DataLoader( datasets, batch_size= batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True)


    return train_loader, valid_loader








if __name__ == '__main__':
    rospy.init_node('pcl2_pub_example', anonymous=True)

    data_dir = "/home/anshul/iros_2019/attentional_pointnet/my_dataset/lidar_img/"

    train_loader, valid_loader =  get_data_loaders(data_dir)
    for batch_idx, (data, img_data, labels) in enumerate(train_loader):
        B = data.shape[0] # Batch size
        N = data.shape[1] # Num of points in PointCloud
        color = np.array([1,1,1])
        color = np.vstack([color]*N)
        data = data.float()

        trans = labels[:,0,:3].float()
        trans = trans.unsqueeze(1)
        rot = -labels[:,0, 3].float()


        # # translate the pointcloud
        data_trans = torch.sub(data, trans)

        rotation_matrix = torch.eye(3).view(1,-1).repeat(B,1)
        rotation_matrix[:, 0] = torch.cos(rot)
        rotation_matrix[:, 1] = -torch.sin(rot)
        rotation_matrix[:, 3] = torch.sin(rot)
        rotation_matrix[:, 4] = torch.cos(rot)
        rotation_matrix = rotation_matrix.view(-1,3,3)
        data_trans = torch.bmm(data_trans, rotation_matrix.transpose(1, 2))

        for i in range(B):
            ros_pub(data[i], "/all_points", color)

            # target = labels[i].float()
            # for x in range(len(target)):
            #     ros_pub_marker(target[x].numpy())
            #     pdb.set_trace()

            pdb.set_trace()
            ros_pub(data_trans[i], "/all_points", color)
            pdb.set_trace()











        #
        # points = data[batch_idx].float()
        # target = labels[batch_idx].float()
        # trans = target[0,:3]
        # # print(target[0,:3])
        # print(points.shape)
        # ros_pub(points, "/all_points", color)
        #
        # # for x in range(len(target)):
        # #     ros_pub_marker(target[x].numpy())
        # #     pdb.set_trace()
        #
        # pdb.set_trace()
        # # translate the pointcloud
        # points = torch.sub(points, trans)
        # print(points.shape)
        # # rotate the pointcloud
        # rot = -target[0,3]
        # rot_mat = torch.eye(3).view(-1)
        # rot_mat[ 0] = torch.cos(rot)
        # rot_mat[ 1] = -torch.sin(rot)
        # rot_mat[ 3] = torch.sin(rot)
        # rot_mat[ 4] = torch.cos(rot)
        # rot_mat = rot_mat.view(3,3)
        # points = torch.mm(points, rot_mat.transpose(0, 1))
        #
        # ros_pub(points, "/all_points", color)
        # pdb.set_trace()
        #









































# class acfr_scan(data.Dataset):
#     def __init__(self, velodyne_dir, num_points=2048, mode='train'):
#         data_dir = "/home/anshul/inria_thesis/datasets/kitti/data_object_velodyne/training/velodyne/000044.bin"
#         calib_dir ="/home/anshul/inria_thesis/datasets/kitti/data_object_velodyne/training/calib/000044.txt"
#         label_dir = "/home/anshul/inria_thesis/datasets/kitti/data_object_velodyne/training/label_2/000044.txt"
#
#         scan = np.fromfile(data_dir, dtype=np.float32).reshape(-1, 4)
#         scan = scan[:, :3] # exclude luminance
#         calib = np.genfromtxt(calib_dir, delimiter=' ',skip_header = 5)
#         calib = calib[0,1:].reshape(-1, 4)
#         # new = np.array([0,0,0,1])
#         calib = np.vstack((calib,[0,0,0,1]))
#         calib = inv(calib)
#         labels = np.genfromtxt(label_dir, delimiter=' ', dtype=None)
#
#         loc = check_for_car(labels,calib,0,10,0,10)
#         print(loc)
#
#
#
#         # rot = np.eye(3,3)
#         # rot[1,1] = -1
#         # rot[2,2] = -1
#         # scan = np.dot(scan, rot)
#         # scan[:,2] = scan[:,2]+2.5
#         cloud = pcl.PointCloud()
#         #create pcl from points
#         cloud.from_array(scan)
#         # print(cloud.size)
#         clipper = cloud.make_cropbox()
#         clipper.set_MinMax(0, 0, -3, 0, 10, 10, 3,0)
#         out_cloud = clipper.filter()
#         vox = out_cloud.make_voxel_grid_filter()
#         vox.set_leaf_size(0.2,0.2,0.2)
#         out_cloud = vox.filter()
#
#         out_cloud = np.asarray(out_cloud)
#         print(out_cloud.shape)
#         # pdb.set_trace()
#
#
#
#         # print (d.shape)
#         # B = scan.shape[0] # Batch size
#         N = out_cloud.shape[0] # Num of points in PointCloud
#         color = np.array([1,1,1])
#         color = np.vstack([color]*N)
#         ros_pub(out_cloud, "/all_points", color)
#         ros_pub_marker(*loc)
#
#
#
#
#
#
#
#
# class acfr_dataset(data.Dataset):
#     def __init__(self, data_dir, num_points=2048, mode='train'):
#         self.npoints = num_points
#         files = list_files(data_dir,"csv")
#         dict = {'4wd': [0,1],'car':[0,1],'ute':[0,1],'van':[0,1],'building':[1,4],'pedestrian':[2,1],'traffic_lights':[3,1],'traffic_sign':[3,1],'tree':[4,2],
#         'trunk':[5,2], 'truck':[6,4],'pole':[7,2],'post': [7,2], 'pillar':[8,3], 'bus': [9,4]}
#         self.data = []
#         self.labels = []
#         for f in files:
#             cat = f.split(".")[0]
#             if dict.has_key(cat):
#                 point_dir = os.path.join(data_dir,f)
#                 d = np.genfromtxt(point_dir, delimiter=',')
#                 d = d[:, [3, 4, 5]]
#                 choice = np.random.choice(len(d), self.npoints, replace=True)
#                 # resample
#                 point_set = d[choice, :]
#                 for i in range(dict.get(cat)[1]):
#                     self.data.append(point_set)
#                     self.labels.append(dict.get(cat)[0])
#         self.data = torch.from_numpy(np.stack(self.data)).float()
#         # self.labels = torch.from_numpy(np.stack(self.labels)).float()
#         self.labels = torch.ByteTensor(np.stack(self.labels))
#
#     def __getitem__(self, index):
#         return self.data[index], self.labels[index]
#
#
#     def __len__(self):
#         # print(len(self.data))
#         return len(self.data)
