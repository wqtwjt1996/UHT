from dataset.coco_text_master import coco_text
ct = coco_text.COCO_Text('coco_text_master/COCO_Text.json')
ct.info()

imgs = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible'),('class','machine printed')])
anns = ct.getAnnIds(imgIds=ct.val, catIds=[('legibility','legible'),('class','machine printed')], areaRng=[0,200])
dataDir='/home/andrew/Documents/Dataset/icdar2017'
dataType='train2014'
dataType_img='train2014_img'
dataType_txt='train2014_gt'

# %matplotlib inline
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import time
import os
pylab.rcParams['figure.figsize'] = (10.0, 8.0)


# get all images containing at least one instance of legible text
imgIds = ct.getImgIds(imgIds=ct.train, catIds=[('legibility','legible')])
for i in range(len(imgIds)):
    img = ct.loadImgs(imgIds[i])[0]
    name = str(10000 + i)
    I = cv2.imread('%s/%s/%s' % (dataDir, dataType, img['file_name']))
    print('/%s/%s' % (dataType, img['file_name']))
    txtPath = os.path.join(dataDir, dataType_txt, name + '.txt')
    gt_f = open(txtPath, 'w')
    annIds = ct.getAnnIds(imgIds=img['id'])
    anns = ct.loadAnns(annIds)
    for ele in anns:
        x1, y1, x2, y2, x3, y3, x4, y4 = int(ele['polygon'][0]), int(ele['polygon'][1]), \
                                         int(ele['polygon'][2]), int(ele['polygon'][3]), \
                                         int(ele['polygon'][4]), int(ele['polygon'][5]), \
                                         int(ele['polygon'][6]), int(ele['polygon'][7])
        string = str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ',' + str(x3) + ',' + str(y3) + ',' + str(x4) + ',' + str(y4) + '\n'
        gt_f.write(string)
    gt_f.close()
    imgPath = os.path.join(dataDir, dataType_img, name + '.jpg')
    cv2.imwrite(imgPath, I)