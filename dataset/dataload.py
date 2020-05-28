import copy
import cv2
import torch.utils.data as data
import scipy.io as io
import numpy as np
from PIL import Image
from util.misc import find_bottom, find_long_edges, split_edge_seqence, get_n_parts, norm2
from bresenham import bresenham
from skimage import exposure
from config import *

global cfg
cfg = init_cfg()

def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image


class TextInstance(object):

    def __init__(self, points, orient, text):
        cv2.setNumThreads(0)
        self.orient = orient
        self.text = text

        remove_points = []

        if len(points) > 4:
            # remove point if area is almost unchanged after removing it
            ori_area = cv2.contourArea(points)
            for p in range(len(points)):
                # attempt to remove p
                index = list(range(len(points)))
                index.remove(p)
                area = cv2.contourArea(points[index])
                if np.abs(ori_area - area) / ori_area < 0.017 and len(points) - len(remove_points) > 4:
                    remove_points.append(p)
            self.points = np.array([point for i, point in enumerate(points) if i not in remove_points])
        else:
            self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)

    def disk_cover(self, unit_disk):
        n_disk = get_n_parts(self.points, self.bottoms, self.e1, self.e2, unit_disk)
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # inverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center
        radii = norm2(inner_points1 - center_points, axis=1)  # disk radius

        return inner_points1, inner_points2, center_points, radii

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TextDataset(data.Dataset):

    def __init__(self, transform):
        super().__init__()

        cv2.setNumThreads(0)
        self.transform = transform

    def parse_mat(self, mat_path):
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4: # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            polygon.append(TextInstance(pts, ori, text))
        return polygon

    def fill_one_polygon(self, mask, point1, point2, r, value):
        try:
            list_points = list(bresenham(int(point1[0]), int(point1[1]), int(point2[0]), int(point2[1])))
        except:
            return np.zeros_like(mask)
        for e in list_points:
            if(int(r) == 0):
                continue
            gaussian_map = exposure.rescale_intensity(cv2.getGaussianKernel(ksize=2 * int(r), sigma=25), out_range=(0, value))
            current_mask = np.zeros(mask.shape[:2], np.float32)
            for g in range(int(gaussian_map.shape[0] / 2)):
                cv2.circle(current_mask, (e[0], e[1]), radius=int(r) - g, color=gaussian_map[g], thickness=-1)
            mask = np.maximum(mask, current_mask)
        return mask

    def make_heatmap(self, center_line, radius, mask, shrink=2):

        for i in range(shrink, len(center_line) - 1 - shrink):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            mask = self.fill_one_polygon(mask, c1, c2, radius[i], value=1).copy()
        return mask

    def get_training_data(self, image, polygons, image_id, image_path):

        try:
            H, W, C = image.shape
        except:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            H, W, C = image.shape

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        if self.transform:
            if(C == 3):
                image, polygons = self.transform(image, copy.copy(polygons))
            elif(C == 4):
                image = image[:,:,:3]
                image, polygons = self.transform(image, copy.copy(polygons))

        reg_mask = np.zeros(image.shape[:2], np.float32)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, radius = polygon.disk_cover(unit_disk=cfg.unit_disk)
                reg_mask = self.make_heatmap(center_points, radius, reg_mask).copy()

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        points = np.zeros((cfg.max_annotation, cfg.max_points, 2))
        length = np.zeros(cfg.max_annotation, dtype=int)

        for i, polygon in enumerate(polygons):
            pts = polygon.points
            points[i, :pts.shape[0]] = polygon.points
            length[i] = pts.shape[0]

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'annotation': points,
            'n_annotation': length,
            'Height': H,
            'Width': W
        }
        return image, reg_mask, meta

    def get_test_data(self, image, image_id, image_path):
        H, W, _ = image.shape

        if self.transform:
            image, polygons = self.transform(image)

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)

        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, meta

    def __len__(self):
        raise NotImplementedError()