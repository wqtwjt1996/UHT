import numpy as np
import cv2
import os
from config import *
from skimage import exposure

global init_cfg
cfg = init_cfg()


def visualize_network_output(img, output, reg_mask, mode='train'):

    vis_dir = os.path.join(cfg.visualization_directory, cfg.dataset_name + '_' + cfg.backbone + '_' + mode)
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)

    prediction = output.permute(0, 2, 3, 1).clone().detach()
    img = img.permute(0, 2, 3, 1).clone()

    img = img.cpu().numpy()
    prediction = prediction.cpu().numpy()
    target = reg_mask.cpu().numpy()

    for i in range(len(output)):
        image = exposure.rescale_intensity(img[i], out_range=(0, 255))
        pred = prediction[i]
        pred = exposure.rescale_intensity(pred, out_range=(0, 255))
        pred = pred.astype(np.uint8)
        targ = (target[i] * 255).astype(np.uint8)

        pred_color = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        targ_color = cv2.applyColorMap(targ, cv2.COLORMAP_JET)

        tcl_show = np.concatenate([image, pred_color, targ_color], axis=1)
        show = cv2.resize(tcl_show, (1024 + 512, 512))

        path = os.path.join(vis_dir, '{}.png'.format(i))
        cv2.imwrite(path, show)

def visualize_detection(image, contours, reg_mask):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    if cfg.bounding_shape:
        image_show = cv2.polylines(image_show, contours, True, (50, 20, 255), 2)
    else:
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i].astype(int))
            cv2.rectangle(image_show, (x, y), (x + w, y + h), (50, 20, 255), 2)
    if (reg_mask is not None):
        reg = (reg_mask * 255).astype(np.uint8)
        reg = cv2.cvtColor(reg, cv2.COLOR_GRAY2BGR)
        reg = cv2.applyColorMap(reg, cv2.COLORMAP_JET)
        image_show = np.concatenate([image_show, reg], axis=1)
        return image_show
    else:
        return image_show

def visualize_detection_end_to_end(image, contours, aster_text, reg_mask):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i].astype(int))
        cv2.rectangle(image_show, (x, y), (x + w, y + h), (50, 20, 255), 2)
        cv2.putText(image_show, aster_text[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 0), 2)
    if (reg_mask is not None):
        tcl = (reg_mask * 255).astype(np.uint8)
        tcl = cv2.cvtColor(tcl, cv2.COLOR_GRAY2BGR)
        tcl = cv2.applyColorMap(tcl, cv2.COLORMAP_JET)
        image_show = np.concatenate([image_show, tcl], axis=1)
        return image_show
    else:
        return image_show
