from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from aster_models.init import create
from aster_models.attention_recognition_head import AttentionRecognitionHead
from aster_models.loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from aster_models.tps_spatial_transformer import TPSSpatialTransformer
from aster_models.stn_head import STNHead

from util.labelmaps import get_vocabulary, labels2strs

# from config import get_args
# global_args = get_args(sys.argv[1:])


class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos, STN_ON=False):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.tps_inputsize = [32, 64]

    self.encoder = create(self.arch,
                      with_lstm=True,
                      n_group=1)
    encoder_out_planes = self.encoder.out_planes

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)
    # self.rec_crit = SequenceCrossEntropyLoss()

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple([32, 100]),
        num_control_points=20,
        margins=[0.05, 0.05])
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=20,
        activation=None)

  def forward(self, input_img):
    return_dict = {}
    return_dict['losses'] = {}
    return_dict['output'] = {}

    x = input_img

    # rectification
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    encoder_feats = self.encoder(x)
    encoder_feats = encoder_feats.contiguous()

    rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, 5, self.eos)

    return rec_pred

if __name__ == "__main__":
  x = torch.randn(3, 3, 32, 100)
  voc = get_vocabulary('ALLCASES_SYMBOLS', EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN')
  net = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=100,
                       sDim=512, attDim=512, max_len_labels=100,
                       eos=dict(zip(voc, range(len(voc))))['EOS'], STN_ON=True)
  # net = ResNet_ASTER(use_self_attention=True, use_position_embedding=True)
  res = net(x)
  print(res.size())

# x = torch.randn(3, 3, 32, 100)
# voc = get_vocabulary('ALLCASES_SYMBOLS', EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN')
# net = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=100,
#                        sDim=512, attDim=512, max_len_labels=100,
#                        eos=dict(zip(voc, range(len(voc))))['EOS'], STN_ON=True)
# # net = ResNet_ASTER(use_self_attention=True, use_position_embedding=True)
# res = net(x)
# print(res.size())