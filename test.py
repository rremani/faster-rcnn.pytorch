# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
print('All loaded')


# XXX
import matplotlib.pyplot as plt
%matplotlib inline

# List of trained models

print(os.listdir('trained-models/res101/table/'))

#### Load model

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
    im (ndarray): a color image in BGR order
    Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
#     im_orig -= cfg.PIXEL_MEANS
#     im_orig -=np.array([[[0,0,0]]])
    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    
    processed_ims = []
    im_scale_factors = []
    
    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        
        im = cv2.resize(im_orig,None,fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)
def load_model(s,ep,x,path='trained-models/res101/table/',net='res101'):
    # What type of model to load? Initializing object of the model.
    # Model path
    load_name = os.path.join(path,'faster_rcnn_{}_{}_{}.pth'.format(1, 12,1955))
    

    # initilize the network here.
    if net == 'vgg16':
        fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif net == 'res101':
        fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=False)
    elif net == 'res50':
        fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif net == 'res152':
        fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    # Creating Architecture
    fasterRCNN.create_architecture()
    # Loading the model
    # checkpoint = torch.load(load_name)
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']
    return fasterRCNN


def data_init(C=1):
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if C > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data, volatile=True)
    im_info = Variable(im_info, volatile=True)
    num_boxes = Variable(num_boxes, volatile=True)
    gt_boxes = Variable(gt_boxes, volatile=True)
    return im_data,im_info,num_boxes,gt_boxes


def forward(im_in,model):
    # Initializing initial tensors
    im_data,im_info,num_boxes,gt_boxes = data_init(C=1)
    
    if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)

    # rgb -> bgr
    im = im_in[:,:,::-1]

    blobs, im_scales = _get_image_blob(im)
   
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
    im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
    gt_boxes.data.resize_(1, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()

    # Forward pass
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label = model(im_data, im_info, gt_boxes, num_boxes)

    # Predictions
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    return scores,boxes

def normalized_predictions(scores,boxes,img2,thresh=0.6,NMS=0.3):
    if cfg.TEST.BBOX_REG:
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
        box_deltas = box_deltas.view(1, -1, 4 * len(classes))
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        
    pred_boxes /= im_scales[0]
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()
    
    
    im2show = np.copy(img2)

    # Iterating over all classes

    for j in range(1, len(classes)):
        
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        
          # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)

            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_dets,NMS, force_cpu=not cfg.USE_GPU_NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            im2show = vis_detections(im2show, classes[j], cls_dets.cpu().numpy(), 0.5)
    return im2show,cls_dets.cpu().numpy()

# Configuration specific initializations
classes = np.asarray(['__background__','table'])
cfg.USE_GPU_NMS = True
torch.device = 'cuda'
max_per_image = 10
thresh = 0.5
# setting NMS threshold
NMS = 0.3

# Loading the model weights
model = load_model(1,12,1955)
# Cuda enabled
model.cuda()
# Enabiling evaluation mode
model.eval()

# Loading the image for which boxes have to be calculated
img = cv2.imread('images/BJAM-1.jpg')
img2=img.copy()
im_in = cv2.Canny(img,50,100)

scores,boxes=forward(im_in,model)

u,bo=normalized_predictions(scores,boxes,img2,thresh,NMS)

bo.shape