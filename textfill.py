import numpy as np
import cv2
from config import *
from get_recognition_result import getStr

global cfg
cfg = init_cfg()


class TextFill(object):

    def __init__(self, model):
        self.model = model

        # evaluation mode
        model.eval()

    def fill(self, data, template, start_coords, fill_value):
        xsize, ysize = data.shape
        top_value = template[start_coords[0], start_coords[1]]
        orig_value = data[start_coords[0], start_coords[1]]
        Height_flow = template[start_coords[0], start_coords[1]] * 0.5
        prob_value = top_value

        stack = set(((start_coords[0], start_coords[1]),))
        if fill_value == orig_value:
            raise ValueError("Filling region with same value "
                             "already present is unsupported. "
                             "Did you already fill this region?")

        while stack:
            x, y = stack.pop()

            if data[x, y] == orig_value:
                data[x, y] = fill_value
                Flow_end = cfg.textfill_flow_end
                if (x > 0 and template[x - 1, y] > Flow_end and template[x - 1, y] <= template[x, y]) \
                        or (x > 0 and template[x - 1, y] >= Height_flow):
                    prob_value = max(prob_value, template[x - 1, y])
                    stack.add((x - 1, y))
                if (x < (xsize - 1) and template[x + 1, y] > Flow_end and template[x + 1, y] <= template[x, y]) \
                        or (x < (xsize - 1) and template[x + 1, y] >= Height_flow):
                    prob_value = max(prob_value, template[x + 1, y])
                    stack.add((x + 1, y))
                if (y > 0 and template[x, y - 1] > Flow_end and template[x, y - 1] <= template[x, y]) \
                        or (y > 0 and template[x, y - 1] >= Height_flow):
                    prob_value = max(prob_value, template[x, y - 1])
                    stack.add((x, y - 1))
                if (y < (ysize - 1) and template[x, y + 1] > Flow_end and template[x, y + 1] <= template[x, y]) \
                        or (y < (ysize - 1) and template[x, y + 1] >= Height_flow):
                    prob_value = max(prob_value, template[x, y + 1])
                    stack.add((x, y + 1))

        return prob_value

    def postprocessing(self, img, mask):
        core_mask = np.zeros_like(mask)
        core_mask[np.where(mask >= cfg.textfill_top)] = 255
        show_mask = core_mask.copy()
        nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(core_mask.astype(np.uint8), connectivity=4)
        show_mask = cv2.cvtColor(show_mask, cv2.COLOR_GRAY2BGR)

        Polygon = []
        TextContent = []
        for i in range(1, centroids.shape[0]):
            draw_mask = np.zeros_like(mask).astype(np.uint8)
            cv2.circle(show_mask, (int(centroids[i][0]), int(centroids[i][1])), 1, (255, 180, 0), -1)
            cv2.putText(show_mask, str(i), (int(centroids[i][0]), int(centroids[i][1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 128, 255), 1)
            cv2.rectangle(show_mask, (stats[i][0], stats[i][1]), (stats[i][0] + stats[i][2], stats[i][1] + stats[i][3]), (150, 50, 255), 1)
            # initial one single text region
            if (labels[int(centroids[i][1])][int(centroids[i][0])] != 0):
                self.fill(draw_mask, mask, (int(centroids[i][1]), int(centroids[i][0])), 200)
            else:
                for x_c in range(stats[i][0], stats[i][0] + stats[i][2]):
                    if (labels[int(centroids[i][1])][x_c] != 0):
                        self.fill(draw_mask, mask, (int(centroids[i][1]), x_c), 200)
                        break
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            draw_mask = cv2.dilate(cv2.erode(draw_mask, kernel), kernel)

            # calculating kernel
            ret, thresh = cv2.threshold(draw_mask, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            Area = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                Area.append(area)
            try:
                k1 = int(max(Area) / 750) + 8
                if(max(Area) >= 20000):
                    k1 = 35
                bigger_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k1, k1))
            except:
                bigger_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

            # get final text region
            draw_mask = cv2.dilate(draw_mask, bigger_kernel)

            ret, thresh = cv2.threshold(draw_mask, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if(cv2.contourArea(cnt, oriented=True) > 0):
                    cnt = cnt[::-1]
                approx = cv2.approxPolyDP(cnt, 1, True)
                e = approx.squeeze(1)
                index = True
                for ele in Polygon:
                    if (e.tolist() == ele.tolist()):
                        index = False
                if(index):
                    Polygon.append(e)
                    if (cfg.spotter):
                        x, y, w, h = cv2.boundingRect(e)
                        TextContent.append(getStr(img[y:y+h, x:x+w])[0].lower())

                assert approx.squeeze(1).shape[1] == 2
        return Polygon, TextContent

    def detect(self, image):
        output = self.model(image)
        image = image[0].data.cpu().numpy()
        pred_mask = output[0, :].data.cpu().numpy().transpose(1, 2, 0)
        pred_mask = pred_mask.reshape((pred_mask.shape[0], pred_mask.shape[1]))
        pred_mask[np.where(pred_mask < 0.0)] = 0.0
        pred_mask[np.where(pred_mask > 1.0)] = 1.0
        Polygon, TextContect = self.postprocessing((image.transpose(1, 2, 0) * cfg.stds + cfg.means) * 255.0, pred_mask)
        output = {
            'image': image,
            'reg': pred_mask
        }
        return Polygon, TextContect, output
