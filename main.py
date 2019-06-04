#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



from __future__ import print_function


import argparse
import os
import shutil
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from modules import recurrent_attention_Loss
from model import AttentionalPointnet
from kitti_custom.kitti_LidarImg_data_provider_v2 import get_data_loaders
from tools.utils import save_checkpoint, AverageMeter, binary_accuracy, percent_overlap, bbox_iou





use_cuda = torch.cuda.is_available()

if use_cuda:
    print('setting gpu on gpu_id: 0') #TODO: find the actual gpu id being used


parser = argparse.ArgumentParser()

# specify data and datapath
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-s', '--save_checkpoints', dest='save_checkpoints', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', default=120, type=int,
                    help='number of total epochs to run')
parser.add_argument('--num_glimpses', default=6, type=int,help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, help='number  epochs to start from')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')

args = parser.parse_args()


data_dir = "/home/anshul/iros_2019/attentional_pointnet/my_dataset/corrected_data/"
train_loader, valid_loader =  get_data_loaders(data_dir)






model = AttentionalPointnet(N = 4096)
if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
criterion = recurrent_attention_Loss().cuda()
seq_len = 3 # for training we keep the sequence length 1 more the maximum number of the objects in one cropped region

def train(epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for batch_idx, (data, img_data, labels) in enumerate(train_loader):

        data_time.update(time.time() - end) # measure data loading time
        B = data.shape[0] # Batch size
        N = data.shape[1] # Num of points in PointCloud

        data, labels, img_data  = data.float(), labels.float(), img_data.float()


        if use_cuda:
            labels, data, img_data =  labels.cuda(), data.cuda(), img_data.cuda()

        img_data  = img_data.unsqueeze(1)

        hidden = torch.zeros(1,B,512).cuda() # initialising the hidden variable for GRU

        optimizer.zero_grad()

        output = model(data, img_data, hidden, seq_len) # (B,4)
        loss = criterion(output, labels, seq_len)

        loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        hidden = hidden.detach()
        losses.update(loss.item(), B)

        pred = output[0]
        prec1 = binary_accuracy(pred[0], labels[:,0,8])
        top1.update(prec1, B)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    return losses.avg



def validate():

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    TP = torch.zeros(0) # True Positives
    CS = torch.zeros(0) # Cosine Similarity

    # switch to evaluate mode
    model.eval()
    # if args.evaluate:
    #     model.train()
    with torch.no_grad():
        end = time.time()

        for batch_idx, (data, img_data, labels) in enumerate(valid_loader):

            B = data.shape[0] # Batch size
            N = data.shape[1] # Num of points in PointCloud

            data, labels, img_data  = data.float(), labels.float(), img_data.float()
            # labels = labels.permute(1,0,2) #(seq,B,5)

            if use_cuda:
                labels, data, img_data =  labels.cuda(), data.cuda(), img_data.cuda()

            img_data  = img_data.unsqueeze(1)
            hidden = torch.zeros(1,B,512).cuda() # initialising the hidden variable for GRU

            optimizer.zero_grad()

            output = model(data, img_data, hidden, seq_len) # (B,4)
            loss = criterion(output, labels, seq_len)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            hidden = hidden.detach()
            losses.update(loss.item(), B)


            pred = output[0]
            prec1 = binary_accuracy(pred[1], labels[:,1,8])
            top1.update(prec1, B)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()



###################################### Final Evaluation ####################################################


            score_seq, loc_seq, box_seq = output


            trans_mat_1 = torch.eye(3).view(1,-1).repeat(seq_len,B,1).cuda()
            trans_mat_1[:,:,0] = loc_seq[:,:,0] # c
            trans_mat_1[:,:,1] = -loc_seq[:,:,1] # -s
            trans_mat_1[:,:,3] = loc_seq[:,:,1] # s
            trans_mat_1[:,:,4] = loc_seq[:,:,0] # c
            trans_mat_1[:,:,2] = loc_seq[:,:,2] #tx
            trans_mat_1[:,:,5] = loc_seq[:,:,3] #ty
            trans_mat_1 = trans_mat_1.view(seq_len*B,3,3)




            trans_mat_2 = torch.eye(3).view(1,-1).repeat(seq_len,B,1).cuda()
            trans_mat_2[:,:,0] = box_seq[:,:,0] # c
            trans_mat_2[:,:,1] = -box_seq[:,:,1] # -s
            trans_mat_2[:,:,3] = box_seq[:,:,1] # s
            trans_mat_2[:,:,4] = box_seq[:,:,0] # c
            trans_mat_2[:,:,2] = box_seq[:,:,2] #tx
            trans_mat_2[:,:,5] = box_seq[:,:,3] #ty
            trans_mat_2 = trans_mat_2.view(seq_len*B,3,3)


            resultant_trans = torch.bmm(trans_mat_1, trans_mat_2)
            resultant_trans = resultant_trans.view(seq_len,B, 9)

            final_trans_params = resultant_trans[:,:,[0,3,2,5]]
            z = (loc_seq[:,:,4] + box_seq[:,:,4]).view(seq_len,B,-1)
            final_trans_params = torch.cat((final_trans_params,z),2)

            loc = final_trans_params[:,:,2:5]
            theta = torch.atan2(final_trans_params[:,:,1], final_trans_params[:,:,0])
            size = box_seq[:,:,5:]

            for a in range(B):
                car_list = check_for_car(labels[a])

                detections = []

                for i in range(seq_len):

                    trans_params = torch.cat((loc[i,a], theta[i,a].view(1), size[i,a]),0)

                    if((score_seq[i,a] > 0.7)):
                        detections.append(trans_params.cpu().numpy())

                TP_region, CS_region = eval_detect_in_region(car_list, detections)
                TP = torch.cat((TP,TP_region),0)
                CS = torch.cat((CS,CS_region),0)










            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       batch_idx, len(valid_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))



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


    return losses.avg, recall, AOS



best_recall = 0

def main():
    # rospy.init_node('pcl2_pub_example', anonymous=True)
    global args, best_recall
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_recall = checkpoint['best_recall']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))



    if args.evaluate:
        validate()
        return


    for epoch in range(args.start_epoch, args.epochs):
        # pdb.set_trace()

        # adjust_learning_rate(optimizer, epoch)
        loss_t = train(epoch)

        # evaluate on validation set
        loss_v, recall, AOS = validate()


        with open("convergence.txt", "a") as myfile:
            myfile.write("{},{},{},{},{}".format(epoch, loss_t, loss_v, recall, AOS)+"\n")


        if (epoch > 40):
            args.lr = 0.001

        if (epoch > 80):
            args.lr = 0.0001

        if (args.save_checkpoints):
            # remember best prec@1 and save checkpoint
            is_best = recall > best_recall
            best_recall = max(recall, best_recall)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_recall': best_recall,
                'optimizer' : optimizer.state_dict(),
            }, is_best)








def check_for_car(label):
    label = label.cpu().numpy()
    car_list = []
    for loc in label:
        if (loc[8] == 1):
            car_list.append([loc[0] , loc[1], loc[2], loc[3], loc[5], loc[6],loc[7]])
    return car_list


def eval_detect_in_region(car_list, detections):
    num_cars = len(car_list)
    num_detect = len(detections)
    TP_region = torch.zeros(num_cars)
    CS_region = torch.zeros(num_detect)
    for i in range(num_cars):
        for j in range(num_detect):
            # IoU = bbox_iou(car_list[i],detections[j])
            IoU, overlap = percent_overlap(car_list[i],detections[j])
            if(overlap >= 0.7):
                TP_region[i] = 1
                CS_region[j] = (1+math.cos(car_list[i][3] - detections[j][3]))/2.0
    return TP_region,CS_region







if __name__ == '__main__':
    main()
