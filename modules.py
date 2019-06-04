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


This code contains the pytorch implementation of pointwise convolution operator.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math


from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment




def get_box3d_corners(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """

    B = centers.shape[0]
    h = sizes[:,0] # (B,1)
    w = sizes[:,1]
    l = sizes[:,2]

    x_corners = torch.stack([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], dim = 1).unsqueeze(2) # (B,8)
    z_corners = torch.stack([h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2], dim =1).unsqueeze(2) # (B,8)
    y_corners = torch.stack([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], dim=1).unsqueeze(2) # (B,8)
    corners = torch.cat((x_corners,y_corners, z_corners),2) # (B,8,3)

    # pdb.set_trace()
    c = headings[:,0] #(B)
    s = headings[:,1]
    trans_params = torch.stack([c, -s, s, c],1) #(B,4)

    rot_mat = torch.eye(3).view(1,-1).repeat(B,1).cuda()
    rot_mat[:,[0,1,3,4]] = torch.squeeze(trans_params)
    rot_mat = rot_mat.view(B,3,3)

    # rotate the pointcloud
    corners = torch.bmm(corners, rot_mat.transpose(1, 2))

    # translate the pointcloud
    corners = torch.add(corners, centers.unsqueeze(1))

    return corners











def _out_size(self, input_size, kernel_size, stride = 1, padding = 0, pool = False,  pool_kernel_size = 2):
    out_size = (input_size - kernel_size + 2 * padding)/stride + 1
    # flat_features = output_size * output_size * channel
    if pool:
        out_size = out_size/pool_kernel_size
    return int(out_size)




def hungarian_matching(pred_loc, target_loc):
    # pred_loc = pred_loc.data.cpu().numpy()
    # target_loc = target_loc.data.cpu().numpy()

    cost_matrix = cdist(target_loc, pred_loc)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return col_ind








class recurrent_attention_Loss(torch.nn.Module):

    def __init__(self):
        super(recurrent_attention_Loss,self).__init__()

    def forward(self, output, labels, seq_len): #trans_params = (R00,R01,R10,R11,Tx,Ty,Tz)

        lossMSE = nn.MSELoss()
        lossHuber = nn.SmoothL1Loss()
        lossCrossEntropy  = nn.CrossEntropyLoss()
        lossBinEntropy = nn.BCELoss()

        batch_size = labels.shape[0]
        score_seq, loc_seq, box_seq  = output
        score_seq = score_seq.permute(1,0) #(B,S,1)
        loc_seq = loc_seq.permute(1,0,2) #(B,S,7)
        box_seq = box_seq.permute(1,0,2) #(B,S,7)

        total_loss = 0

        for B in range(batch_size):
            target =  labels[B,:seq_len,8] #(3,1)
            car_loc = torch.squeeze(target.nonzero())
            num_car = car_loc.nelement()


            if(num_car!=0):
                trans_params_hat = torch.zeros(seq_len,5).cuda()
                rot = labels[B,car_loc, 3] # CHANGED removed negative sign
                trans_params_hat[:num_car, 0] = torch.cos(rot)
                trans_params_hat[:num_car, 1] = torch.sin(rot)
                trans_params_hat[:num_car,2:5] = labels[B, car_loc, :3]
                # trans_params_hat[:num_car,5:] = labels[B, car_loc, 5:8]

                if(num_car!=seq_len):
                    zero_rot = torch.tensor(0).float()
                    trans_params_hat[num_car:, 0] = torch.cos(zero_rot)
                    trans_params_hat[num_car:, 1] = torch.sin(zero_rot)
                    trans_params_hat[num_car:,2:5] = torch.tensor([0,0,-4]).cuda()
                    # trans_params_hat[num_car:,5:] = torch.tensor([2,2.5,5]).cuda()

                # if(num_car == 3):
                #     print("yes", B, trans_params_hat)

                ind = hungarian_matching(loc_seq[B,:,2:5].detach().cpu(), labels[B,car_loc,:3].view(num_car,-1).detach().cpu())
                ind = np.concatenate((ind,[i for i in range(seq_len) if i not in ind]))

                # loc_seq_matched = loc_seq[B,ind].view(num_car,-1)
                loc_seq_matched = loc_seq[B,ind]
                score_seq_matched = score_seq[B, ind]
                box_seq_matched = box_seq[B, ind, :5]
                size_seq_matched = box_seq[B, ind, 5:]

                # pos_ind = score_seq_matched > 0.5
                # pos_ind = torch.squeeze(pos_ind.nonzero())
                # num_pos_ind = pos_ind.nelement()
                # pdb.set_trace()



                trans_mat_1 = torch.eye(3).view(1,-1).repeat(seq_len,1).cuda()
                trans_mat_1[:,0] = loc_seq_matched[:,0] # c
                trans_mat_1[:,1] = -loc_seq_matched[:,1] # -s
                trans_mat_1[:,3] = loc_seq_matched[:,1] # s
                trans_mat_1[:,4] = loc_seq_matched[:,0] # c
                trans_mat_1[:,2] = loc_seq_matched[:,2] #tx
                trans_mat_1[:,5] = loc_seq_matched[:,3] #ty
                trans_mat_1 = trans_mat_1.view(seq_len,3,3)


                # trans_mat_1 =  trans_mat_1.detach() # here we detach bcoz we aim to train two network seperately!

                trans_mat_2 = torch.eye(3).view(1,-1).repeat(seq_len,1).cuda()
                trans_mat_2[:,0] = box_seq_matched[:,0] # c
                trans_mat_2[:,1] = -box_seq_matched[:,1] # -s
                trans_mat_2[:,3] = box_seq_matched[:,1] # s
                trans_mat_2[:,4] = box_seq_matched[:,0] # c
                trans_mat_2[:,2] = box_seq_matched[:,2] #tx
                trans_mat_2[:,5] = box_seq_matched[:,3] #ty
                trans_mat_2 = trans_mat_2.view(seq_len,3,3)




                resultant_trans = torch.bmm(trans_mat_1, trans_mat_2)
                resultant_trans = resultant_trans.view(seq_len, 9)

                final_trans_params = resultant_trans[:,[0,3,2,5]]
                z = (loc_seq_matched[:,4] + box_seq_matched[:,4]).view(seq_len,-1)
                final_trans_params = torch.cat((final_trans_params,z),1)

                rotx = loc_seq_matched[:,0]**2 + loc_seq_matched[:,1]**2
                reg_loss_1 = lossMSE(rotx, torch.tensor([1]).cuda().float())

                rotx = final_trans_params[:,0]**2 + final_trans_params[:,1]**2
                reg_loss_2 = lossMSE(rotx, torch.tensor([1]).cuda().float())
                # reg_loss = reg_loss.mean()


                loss_where = lossHuber(loc_seq_matched , trans_params_hat)
                loss_what = lossBinEntropy(score_seq_matched, target)
                loss_residual = lossHuber(final_trans_params[car_loc].view(num_car,-1) , trans_params_hat[car_loc].view(num_car,-1))
                loss_size = lossHuber(size_seq_matched[car_loc].view(num_car,-1), labels[B,:num_car,5:8].view(num_car,-1))

                total_seq_loss = loss_where + loss_what + 0.01*(reg_loss_1+ reg_loss_2) + 1.5*loss_residual + 0.5*loss_size
            else:
                loss_what = lossBinEntropy(score_seq[B], target)
                total_seq_loss = loss_what

            total_loss += total_seq_loss
        return total_loss/batch_size















class point_context_network(nn.Module):
    def __init__(self, num_points = 4096,out_size = 1024):
        super(point_context_network, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.mp1 = torch.nn.MaxPool1d(num_points
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        return x




class img_context_network(nn.Module):

    def __init__(self, input_size = 100, hidden_size = 512): #TODO: repair this
        super(img_context_network, self).__init__()

        self.conv_drop = nn.Dropout2d()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5) #img_size = 96
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding = 0)  #img_size = 24
        img_size = _out_size(self, input_size, kernel_size = 5, padding = 0, pool = True, pool_kernel_size = 4)

        self.conv2 = nn.Conv2d(16, 16, padding = 1, kernel_size=3) #img_size = 24
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)  #img_size = 12
        img_size = _out_size(self, img_size, padding = 1, kernel_size=3, pool = True, pool_kernel_size = 2 )

        self.conv3 = nn.Conv2d(16, 32, padding = 1, kernel_size=3) #img_size = 12
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)  #img_size = 6
        img_size = _out_size(self, img_size, kernel_size=3, padding = 1, pool = True, pool_kernel_size = 2)
        D_in = img_size * img_size * 32
        self.fc1 = nn.Linear(int(D_in), hidden_size)


    def forward(self, phi):

        # Batch Normalise every layer ??
        phi = F.relu(self.pool1(self.conv1_bn(self.conv1(phi))))
        phi = F.relu(self.pool2(self.conv2_bn(self.conv2(phi))))
        phi = F.relu(self.pool3(self.conv3_bn(self.conv3(phi))))

        # Flatten up the image
        phi = phi.view(phi.shape[0], -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        # phi_out = F.normalize(phi_out, p=2, dim=1)

        return phi_out




class location_network(nn.Module):

    def __init__(self, input_size = 512, hidden_size = 256, out_size = 5):
        super(location_network, self).__init__()
        # self.std = std
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128, track_running_stats=False)
        # self.fc2 = nn.Linear(hidden_size,128)
        self.fc3 = nn.Linear(128,out_size)
        self.dropout1 = nn.Dropout(p = 0.2)
        # self.dropout2 = nn.Dropout(p = 0.5)

    def forward(self, ht):
        # ht = self.dropout1(F.relu(self.bn1(self.fc1(ht))))
        ht = F.relu(self.bn1(self.fc1(ht)))
        # ht = F.relu(self.fc1(ht))
        # ht = self.dropout2(F.relu(self.fc2(ht)))
        trans_params = self.fc3(ht)
        return trans_params #(cos,sin,Tx,Ty,Tz)  #,H,W,L )






class STN3d_crop(nn.Module):

    def __init__(self, x = 3, y = 5, z = 3, num_points = 512):
        super(STN3d_crop, self).__init__()
        self.xmin = -x/2.0
        self.xmax =  x/2.0
        self.ymin = -y/2.0
        self.ymax =  y/2.0
        self.zmin = -z/2.0
        self.zmax =  z/2.0
        self.N = num_points
    def forward(self, points_tensor, trans_params): #points_tensor.shape = (B,4096,3) loc.shape = (B,3)
        B = points_tensor.shape[0]
        N = points_tensor.shape[1]

        loc = trans_params[:,2:]
        c = trans_params[:,0]
        s = trans_params[:,1]

        # We transposed the rotation matrix bcoz we want to transform point cloud here
        rot_mat = torch.eye(3).view(1,-1).repeat(B,1).cuda()
        rot_mat[:,0] = c
        rot_mat[:,1] = s
        rot_mat[:,3] = -s
        rot_mat[:,4] = c
        rot_mat = rot_mat.view(B,3,3)


        # translate the pointcloud
        transformed_points = torch.sub(points_tensor, loc.unsqueeze(1))

        # rotate the pointcloud
        transformed_points = torch.bmm(transformed_points, rot_mat.transpose(1, 2)) # transpose here is bcoz of property (AB) = B.T * A.T

        # print(points_tensor)

        # Check the points in the bounding box
        min_t = torch.tensor([self.xmin, self.ymin, self.zmin]).cuda()
        max_t = torch.tensor([self.xmax, self.ymax, self.zmax]).cuda()
        # pdb.set_trace()
        t = torch.sum(min_t < transformed_points, 2) + torch.sum(transformed_points < max_t, 2)
        # print(transformed_points < max_t)
        t = t==6

        batch_crop = []
        for i in range(B):
            ind = torch.squeeze(t[i].nonzero())
            points_count = ind.nelement() # what for zero case??

            if(points_count > 1):
                prob = torch.randperm(len(ind))
                if(len(ind) > self.N):
                    idx = prob[:self.N]
                    crop = transformed_points[i, ind[idx]]
                    # print(len(crop))
                else:
                    r = int(self.N/len(ind))
                    ind = ind.repeat(r+1)
                    crop = transformed_points[i, ind[:self.N]]
                    # print(len(crop))
            else:
                crop = torch.ones(self.N,3).cuda()
            batch_crop.append(crop)
        batch_crop = torch.stack(batch_crop)

        return batch_crop















class PointNetfeat(nn.Module):
    def __init__(self, c_in = 3, num_points = 512, global_feat = True):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(c_in, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, pointfeat], 1)







class RegBoundingBox(nn.Module):
    def __init__(self,c_in = 1024, k = 8):
        super(RegBoundingBox, self).__init__()
        self.fc1 = nn.Linear(c_in, 256)
        # self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x  #(cos,sin,Tx,Ty,Tz,H,W,L)



class SimpleClassifier(nn.Module):
    def __init__(self,c_in = 512, k = 1):
        super(SimpleClassifier, self).__init__()
        # self.fc1 = nn.Linear(c_in, 512)
        self.fc2 = nn.Linear(c_in, 128)
        self.fc3 = nn.Linear(128, k)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(p = 0.2)
        self.dropout2 = nn.Dropout(p = 0.5)
    def forward(self, x):
        # x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout1(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return torch.squeeze(torch.sigmoid(x))
