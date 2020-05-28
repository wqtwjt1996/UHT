import os
import time
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset.total_text import TotalText
from dataset.coco_text import COCO_Text

from network.uht_net import UHT_Net
from textfill import TextFill
from util.augmentation import BaseTransform
from visualize import visualize_detection, visualize_detection_end_to_end
from util.misc import to_device, mkdirs, rescale_result

from config import *

def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])

def write_to_file(contours, aster_text, file_path):
    with open(file_path, 'w') as f:
        for i in range(len(contours)):
            if (cfg.dataset_name == 'total_text'):
                cont = np.stack([contours[i][:, 1], contours[i][:, 0]], 1)
            else:
                cont = np.stack([contours[i][:, 0], contours[i][:, 1]], 1)
            if (not cfg.bounding_shape or cfg.spotter):
                x, y, w, h = cv2.boundingRect(contours[i].astype(int))
                f.write(str(x) + ',' +
                        str(y) + ',' +
                        str(x + w) + ',' +
                        str(y) + ',' +
                        str(x + w) + ',' +
                        str(y + h) + ',' +
                        str(x) + ',' +
                        str(y + h))
                if (cfg.spotter):
                    f.write(',' + aster_text[i] + '\n')
                else:
                    f.write('\n')
            else:
                cont = cont.flatten().astype(str).tolist()
                cont = ','.join(cont)
                if (len(cont.split(',')) >= 6):
                    f.write(cont + '\n')



def inference(detector, test_loader, output_dir):

    total_time = 0.0

    for i, (image, reg_mask, meta) in enumerate(test_loader):

        image, reg_mask = to_device(image, reg_mask)

        torch.cuda.synchronize()
        start = time.time()

        index = 0
        contours, aster_text, output = detector.detect(image)

        torch.cuda.synchronize()
        end = time.time()
        total_time += end - start
        fps = (i + 1) / total_time
        print('detect {} | {} images: {}. ({:.2f} fps)'.format(i + 1, len(test_loader), meta['image_id'][index], fps))

        # visualization
        pred_mask = output['reg']
        img_show = image[index].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        if (cfg.spotter):
            pred_vis = visualize_detection_end_to_end(img_show, contours, aster_text, pred_mask)
        else:
            pred_vis = visualize_detection(img_show, contours, pred_mask)
        gt_contour = []
        for annot, n_annot in zip(meta['annotation'][index], meta['n_annotation'][index]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())
        gt_vis = visualize_detection(img_show, gt_contour, reg_mask[index].cpu().numpy())
        im_vis = np.concatenate([pred_vis, gt_vis], axis=0)
        path = os.path.join(cfg.visualization_directory, '{0}_{1}_test'.format(cfg.dataset_name, cfg.backbone), meta['image_id'][index])
        cv2.imwrite(path.replace('.gif', '.png'), im_vis)

        H, W = meta['Height'][index].item(), meta['Width'][index].item()
        img_show, contours = rescale_result(img_show, contours, H, W)

        mkdirs(output_dir)
        write_to_file(contours, aster_text,
                      os.path.join(output_dir, meta['image_id'][index].replace('ts_', '')
                                   .replace('.jpg', '.txt').replace('.JPG', '.txt').replace('.png', '.txt').replace('.gif', '.txt')))

def main():

    if cfg.dataset_name == "total_text":
        testset = TotalText(
            data_root=cfg.dataset_root,
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
        )
    elif cfg.dataset_name == "coco_text":
        testset = COCO_Text(
            data_root=cfg.dataset_root,
            ignore_list=None,
            is_training=False,
            transform=BaseTransform(size=cfg.img_size, mean=cfg.means, std=cfg.stds)
        )
    else:
        testset = None
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.testing_num_workers)

    # Model
    model = UHT_Net()
    model_path = cfg.evaluation_model_directory
    load_model(model, model_path)

    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True
    detector = TextFill(model)

    print('Start testing Model.')
    output_dir = os.path.join(cfg.prediction_output_directory, cfg.dataset_name + '_' + cfg.backbone)
    inference(detector, test_loader, output_dir)



if __name__ == '__main__':
    from torch.multiprocessing import set_start_method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    global cfg
    cfg = init_cfg()
    vis_dir = os.path.join(cfg.visualization_directory, '{0}_{1}_test'.format(cfg.dataset_name, cfg.backbone))
    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    main()