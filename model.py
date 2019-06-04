#!/usr/bin/env python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import point_context_network,location_network,STN3d_crop, img_context_network, SimpleClassifier,RegBoundingBox, PointNetfeat
# from non_local_simple_version import NONLocalBlock1D



class AttentionalPointnet(nn.Module):

    def __init__(self,N):
        super(AttentionalPointnet, self).__init__()
        hidden_state = 512
        self.context = point_context_network(N,1024)
        self.img_context = img_context_network(120,1024)
        self.rnn = nn.GRU(1024, hidden_state, 1)

        self.classifier = SimpleClassifier(hidden_state, k = 1)

        self.locator = location_network(input_size = hidden_state, hidden_size = 256)
        self.stn = STN3d_crop(x = 3, y = 6, z = 3, num_points = 512)# Dimensions of the bounding box
        self.get_feat = PointNetfeat(c_in = 3, num_points = 512, global_feat= True)
        self.box_reg = RegBoundingBox(c_in = 1024, k = 8) #h,w,l
        self.dropout1 = nn.Dropout(p = 0.3)



    def forward(self, points_tensor, img_tensor, init_hidden,seq_len):

        score_seq = []
        loc_seq = []
        box_seq = []

        l_context = self.context(points_tensor.permute(0,2,1)) # (B,1024)
        i_context = self.img_context(img_tensor) # (B,1024)

        in_seq = l_context + i_context
        in_seq = self.dropout1(in_seq)

        in_seq = in_seq.unsqueeze(0) #(1,B,512)
        in_seq = in_seq.expand(seq_len,-1,-1) #(6,B,512)


        out_seq, hn = self.rnn(in_seq, init_hidden) # out_seq.shape = (6,B,1024)



        for i in range(seq_len):
            prob = self.classifier(out_seq[i]) # (B,1)
            trans_params = self.locator(out_seq[i]) # (B,3)


            attention = self.stn(points_tensor,trans_params) # (B,512,3)
            # pdb.set_trace()
            # attention = attention.detach()
            feat = self.get_feat(attention.permute(0,2,1)) # (B, 1024)
            box_params = self.box_reg(feat) # (B,8)

            score_seq.append(prob)
            loc_seq.append(trans_params)
            box_seq.append(box_params)

        score_seq = torch.stack(score_seq)
        box_seq = torch.stack(box_seq)
        loc_seq = torch.stack(loc_seq)
        return score_seq,loc_seq, box_seq
