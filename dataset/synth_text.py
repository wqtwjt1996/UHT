import scipy.io as io
import numpy as np
import os
import cv2

from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance

class SynthText(TextDataset):

    def __init__(self, data_root, is_training=True, transform=None):
        super().__init__(transform)
        self.data_root = data_root
        self.is_training = is_training

        self.image_root = data_root
        self.annotation_root = os.path.join(data_root, 'gt')

        with open(os.path.join(data_root, 'image_list.txt')) as f:
            self.annotation_list = [line.strip() for line in f.readlines()]

    def cal_S(self, points):
        # print(points)
        area = 0
        q = points[-1]
        for p in points:
            area += p[0] * q[1] - p[1] * q[0]
            q = p
        return area / 2

    def parse_txt(self, annotation_path):

        with open(annotation_path) as f:
            lines = [line.strip() for line in f.readlines()]
            image_id = lines[0]
            polygons = []
            for line in lines[1:]:
                points = [float(coordinate) for coordinate in line.split(',')]
                points = np.array(points, dtype=int).reshape(4, 2)
                polygon = TextInstance(points, 'c', 'abc')
                if(abs(points.max()) <= 1024 and abs(points.min()) <= 1024):
                    polygons.append(polygon)
                # print(self.cal_S(points))

                # print(polygons)
        return image_id, polygons


    def __getitem__(self, item):

        # Read annotation
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        image_id, polygons = self.parse_txt(annotation_path)

        # Read image data
        image_path = os.path.join(self.image_root, image_id)
        # print(image_path)
        image = pil_load_img(image_path)
        # cv2.imwrite('image.jpg', image)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        return self.get_training_data(image, polygons, image_id=image_id, image_path=image_path)

    def __len__(self):
        return len(self.annotation_list)

if __name__ == '__main__':
    import os
    from util.augmentation import BaseTransform, Augmentation

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=512, mean=means, std=stds
    )

    trainset = SynthText(
        data_root='/home/andrew/Documents/Dataset/SynthText/SynthText',
        is_training=True,
        transform=transform
    )

    img, tcl_mask, meta = trainset[28114]
    cv2.imwrite('img.jpg', img.transpose(1, 2, 0))
    # print(tr_mask.min())

    #
    # for idx in range(285143, len(trainset)):
    #     img, train_mask, tr_mask, tcl_mask, meta = trainset[idx]
    #     print(idx, img.shape)
    #     if(idx % 20 == 0):
    #         import cv2
    #         cv2.imwrite('check_data.jpg', tcl_mask * 255.0)