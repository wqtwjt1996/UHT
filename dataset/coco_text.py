import os
import numpy as np

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from config import *

global cfg
cfg = init_cfg()

class COCO_Text(TextDataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root,
                                       'train2014_img' if is_training else 'val2014_img')
        self.annotation_root = os.path.join(data_root,
                                            'train2014_gt' if is_training else 'val2014_gt')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '').replace('.JPG', '').replace('.png', '').replace('.gif', '') not in ignore_list, self.image_list))
        self.annotation_list = ['{}.txt'.format(img_name.replace('.jpg', '').replace('.JPG', '').replace('.png', '').replace('.gif', '')) for img_name in self.image_list]

    def parse_txt(self, txt_path):
        f = open(txt_path, 'r')
        Data = f.readlines()
        polygons = []
        for ele in Data:
            ele = ele.replace('\ufeff', '')
            line_data = ele[:-1].split(',', 8)
            if self.is_training == False:
                line_data = line_data[:-1]
            coord = list(map(int, line_data))
            text = line_data[-1]
            ori = 'c'
            pts = []
            pts.append(np.stack([coord[0], coord[1]]).T.astype(np.int32))
            pts.append(np.stack([coord[2], coord[3]]).T.astype(np.int32))
            pts.append(np.stack([coord[4], coord[5]]).T.astype(np.int32))
            pts.append(np.stack([coord[6], coord[7]]).T.astype(np.int32))
            polygons.append(TextInstance(np.array(pts), ori, text))
        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        image = pil_load_img(image_path)

        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_txt(annotation_path)

        for i, polygon in enumerate(polygons):
            polygon.find_bottom_and_sideline()

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    trainset = COCO_Text(
        data_root='/home/andrew/Documents/Dataset/coco_text',
        ignore_list=None,
        is_training=False,
        transform=Augmentation(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
    )

    for idx in range(0, len(trainset)):
        img, reg_mask, train_mask, meta = trainset[idx]
        print(idx, img.shape)