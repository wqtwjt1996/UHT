import cv2, os
from aster_models.model_builder import ModelBuilder
import util.labelmaps as UL
import torch
import string
import numpy as np
from config import *

def _normalize_text(text):
  text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  return text
  # return text.lower()

def get_str_list(output):
    # label_seq
    # assert output.dim() == 2 and target.dim() == 2

    voc = UL.get_vocabulary('ALLCASES_SYMBOLS', EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN')
    my_char2id = dict(zip(voc, range(len(voc))))
    my_id2char = dict(zip(range(len(voc)), voc))

    end_label = my_char2id['EOS']
    unknown_label = my_char2id['UNKNOWN']
    num_samples, max_len_labels = output.size()
    num_classes = len(my_char2id.keys())
    # assert num_samples == target.size(0) and max_len_labels == target.size(1)
    output = UL.to_numpy(output)
    # target = to_numpy(target)

    # list of char list
    pred_list, targ_list = [], []
    for i in range(num_samples):
        pred_list_i = []
        for j in range(max_len_labels):
            if output[i, j] != end_label:
                if output[i, j] != unknown_label:
                    pred_list_i.append(my_id2char[output[i, j]])
            else:
                break
        pred_list.append(pred_list_i)

    pred_list = [_normalize_text(pred) for pred in pred_list]
    return pred_list

def load_checkpoint(fpath):
    load_path = fpath
    checkpoint = torch.load(load_path)
    return checkpoint

def getStr(img):
    global cfg
    cfg = init_cfg()
    voc = UL.get_vocabulary('ALLCASES_SYMBOLS', EOS='EOS', PADDING='PADDING', UNKNOWN='UNKNOWN')
    ASTER = ModelBuilder(arch='ResNet_ASTER', rec_num_classes=97,
                       sDim=512, attDim=512, max_len_labels=100,
                       eos=dict(zip(voc, range(len(voc))))['EOS'], STN_ON=True)
    checkpoint = load_checkpoint('/home/andrew/Documents/aster.pytorch-master/demo.pth.tar')
    ASTER.load_state_dict(checkpoint['state_dict'])

    ASTER.eval()
    input_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(dim=0)
    input_tensor = 2.0 * input_tensor / 255.0 - 1
    input_tensor = input_tensor.float()
    output = ASTER(input_tensor)
    return get_str_list(output)
