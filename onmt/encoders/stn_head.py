from __future__ import absolute_import

import math
import numpy as np
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class STNHead(nn.Module):
  def __init__(self, in_planes, num_ctrlpoints, activation='none'):
    super(STNHead, self).__init__()

    self.in_planes = in_planes
    self.num_ctrlpoints = num_ctrlpoints
    self.activation = activation

    gfc = 64
    input_width = 192
    input_height = 48
    n_downscale = 3
    gfs_width = input_width // (2**n_downscale)
    gfs_height = input_height // (2**n_downscale)
    loc_conv = []
    loc_conv.append(nn.Conv2d(1, gfc, kernel_size=3, stride=1, padding=1))
    loc_conv.append(nn.BatchNorm2d(gfc))
    loc_conv.append(nn.ReLU(True))
    for i in range(n_downscale):
      loc_conv.append(nn.Conv2d(2**i*gfc, 2**(i+1)*gfc, kernel_size=4, stride=2, padding=1))
      loc_conv.append(nn.BatchNorm2d(2**(i+1)*gfc))
      loc_conv.append(nn.ReLU(True))
      loc_conv.append(nn.Conv2d(2**(i+1)*gfc, 2**(i+1)*gfc, kernel_size=3, stride=1, padding=1))
      loc_conv.append(nn.BatchNorm2d(2**(i+1)*gfc))
      loc_conv.append(nn.ReLU(True))

    self.stn_convnet = nn.Sequential(*loc_conv)


    self.numel = (gfs_width*gfs_height)*gfc*(2**n_downscale)
    self.stn_fc1 = nn.Sequential(
      nn.Linear(self.numel, 1024),
      nn.ReLU(True),
      nn.Linear(1024, 1024),
      nn.ReLU(True),
    )
    self.stn_fc2 = nn.Linear(1024, num_ctrlpoints*2)

    padW = 24
    padH = 6
    h = 36
    w = 144
    target_control_points = torch.Tensor(np.float32(
                        [ [padW, padH], 
                          [padW, h-1+padH], 
                          [w-1+padW, h-1+padH], 
                          [w-1+padW, padH] 
                        ]))

    bias = target_control_points.view(-1)
    self.stn_fc2.bias.data.copy_(bias)
    self.stn_fc2.weight.data.zero_()

    # self.init_weights(self.stn_convnet)
    # self.init_weights(self.stn_fc1)
    # self.init_stn(self.stn_fc2)


  def init_weights(self, module):
    for m in module.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()

  def init_stn(self, stn_fc2):
    margin = 0.01
    sampling_num_per_side = int(self.num_ctrlpoints / 2)
    ctrl_pts_x = np.linspace(margin, 1.-margin, sampling_num_per_side)
    ctrl_pts_y_top = np.ones(sampling_num_per_side) * margin
    ctrl_pts_y_bottom = np.ones(sampling_num_per_side) * (1-margin)
    ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
    ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
    ctrl_points = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0).astype(np.float32)
    if self.activation is 'none':
      pass
    elif self.activation == 'sigmoid':
      ctrl_points = -np.log(1. / ctrl_points - 1.)
    stn_fc2.weight.data.zero_()
    stn_fc2.bias.data = torch.Tensor(ctrl_points).view(-1)

  def forward(self, x):
    x = self.stn_convnet(x)
    x = x.view(-1, self.numel)
    x = self.stn_fc2(self.stn_fc1(x))
    x = x.view(-1, self.num_ctrlpoints, 2)
    # x = self.stn_convnet(x)
    # batch_size, _, h, w = x.size()
    # x = x.view(batch_size, -1)
    # img_feat = self.stn_fc1(x)
    # x = self.stn_fc2(img_feat)
    # if self.activation == 'sigmoid':
    #   x = F.sigmoid(x)
    # x = x.view(-1, self.num_ctrlpoints, 2)
    return x, x
