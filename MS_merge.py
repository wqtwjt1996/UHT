import cv2
import os
from nms import nms as NMS
import datetime, time

import nms.fast as fast
import nms.felzenszwalb as felzenszwalb
import nms.malisiewicz as malisiewicz

idRoot = '/home/andrew/Documents/Dataset/Total-Text/Images/Test'
idList = []
timeList = []
accu_time = 0.0
# print(os.listdir(idRoot))
for ele in os.listdir(idRoot):
    idList.append(int(ele[3:-4]))
for index in idList:
    print(index)
    imgPath = '/home/andrew/Documents/Dataset/Total-Text/Images/Test/img' + str(index) + '.jpg'
    txtPath = '/home/andrew/Documents/TextDetection_CVPR2020/output_total_text/total_text_resnet/img' + str(index) + '.txt'
    desPath = '/home/andrew/Documents/TextDetection_CVPR2020/output/ms_total_text_0308/img' + str(index) + '.txt'

    img = cv2.imread(imgPath)
    Data = open(txtPath, 'r').readlines()
    maxNum = 0
    for ele in Data:
        temp = int((len(ele[:-1].split(',')) - 1) / 2)
        if (maxNum < temp):
            maxNum = temp
    # print('max: ' + str(maxNum))

    Poly = []
    Scores = []
    for ele in Data:
        line = list(map(int, ele[:-1].split(',')[:-1]))
        polygon = []
        X = []
        Y = []
        for i in range(len(line)):
            if (i % 2 == 1):
                X.append(line[i])
            else:
                Y.append(line[i])
        assert len(X) == len(Y)
        if (len(X) < maxNum):
            for i in range(len(X)):
                polygon.append((X[i], Y[i]))
            for i in range(len(X), maxNum):
                polygon.append((X[0], Y[0]))
        else:
            for i in range(len(X)):
                polygon.append((X[i], Y[i]))
        # print(len(polygon))
        Poly.append(polygon)
        Scores.append(float(ele[:-1].split(',')[-1]))

    assert len(Poly) == len(Scores)
    # finalPolys = []
    # for i in range(len(Poly)):
    #     print(Poly[i])
    #     print(Scores[i])
    # print(len(Poly))

    # print(range(len(Data)))
    start = time.time()
    key = NMS.polygons(Poly, Scores, num_algorithm=fast.nms)
    end = time.time()
    accu_time += (end - start)
    print(accu_time)
    wf = open(desPath, 'w')
    for k in key:
        wf.write(",".join(Data[k][:-1].split(',')[:-1]) + '\n')
    wf.close()