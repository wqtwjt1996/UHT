import scipy.io as io
import numpy as np
import os
import re

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class TotalText(TextDataset):

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

        self.image_root = os.path.join(data_root, 'Images', 'Train' if is_training else 'Test')
        self.annotation_root = os.path.join(data_root, 'gt', 'Train' if is_training else 'Test')
        self.image_list = os.listdir(self.image_root)
        i = 0
        while i < len(self.image_list):
            if ('._' in self.image_list[i]):
                self.image_list.remove(self.image_list[i])
            else:
                i += 1
        # print(len(self.annotation_root))
        self.image_list = list(filter(lambda img: img.replace('.jpg', '').replace('.JPG', '') not in ignore_list, self.image_list))
        self.annotation_list = ['poly_gt_{}.mat'.format(img_name.replace('.jpg', '').replace('.JPG', '')) for img_name in self.image_list]

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygons = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0] if len(cell[4]) > 0 else '#'
            ori = cell[5][0] if len(cell[5]) > 0 else 'c'

            if len(x) < 4:  # too few points
                continue
            pts = np.stack([x, y]).T.astype(np.int32)
            polygons.append(TextInstance(pts, ori, text))

        return polygons

    def __getitem__(self, item):

        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)

        # Read image data
        image = pil_load_img(image_path)

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_mat(annotation_path)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = TotalText(
        data_root='/home/andrew/Documents/Dataset/Total-Text',
        # ignore_list='./ignore_list.txt',
        is_training=False,
        transform=transform
    )

    img, reg_mask, train_mask, meta = trainset[291]
    import cv2
    from skimage import exposure
    color_img = np.transpose(img, (1, 2, 0)) * 255
    color_mask = exposure.rescale_intensity(reg_mask, out_range=(0, 255))
    color_mask = color_mask.astype(np.uint8)
    color_mask = cv2.applyColorMap(color_mask, cv2.COLORMAP_JET)

    tcl = np.where(reg_mask >= 0.9, 255, 0)
    tr = np.where(reg_mask > 0.0, 255, 0)

    # print(color_img.shape, reg_mask.shape, color_mask.shape, tcl.shape, tr.shape)
    # print(color_img.max())
    # cv2.imwrite('img641.jpg', color_img)
    # cv2.imwrite('reg_mask.jpg', color_mask)
    # cv2.imwrite('tcl.jpg', tcl)
    # cv2.imwrite('tr.jpg', tr)

    for idx in range(0, len(trainset)):
        img, train_mask, tcl_mask, meta = trainset[idx]
        print(idx, img.shape)