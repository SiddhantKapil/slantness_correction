import cv2
import numpy as np
from collections import deque


class RotateAndDeslantImage:

    def __init__(self):
        pass

    # rotate image if text not horizontal
    def rotate_image(self, im):

        coords = np.column_stack(np.where(im == 0))
        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = im.shape[:2]
        center = (w // 2, h // 2)
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(im, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # find the maximum point which is needed to calculate the slope
    def find_maxima(self, im_, row_start_idx, col_start_idx, col_stop_idx):
        li = im_[row_start_idx - 1, col_start_idx - 4:col_stop_idx + 4]
        next_li = [i for i in range(len(li)) if li[i] != 1]
        if not next_li:
            return row_start_idx, col_start_idx
        row_start_idx, col_start_idx = self.find_maxima(im_, row_start_idx - 1, col_start_idx - 2 + next_li[0],
                                           col_start_idx - 2 + next_li[-1])
        return row_start_idx, col_start_idx

    # deslant image by calculating its slope and then rotating it overcome the effect of that shift
    def deslant_image(self, im, pad_size=25):
        process = True
        p_img = None

        # make border to prevent information loss
        im = cv2.copyMakeBorder(im, 0, 0, pad_size, pad_size, cv2.BORDER_CONSTANT, value=1)

        # start checking for written text after skipping the padded size
        im_ = np.array(im, dtype=np.float64)
        col_start_idx = pad_size
        row_start_idx = im_[:, pad_size].tolist().index(0)
        li = im_[row_start_idx, pad_size:]
        total_cols = 0
        for i in range(len(li)):
            if li[i] != 0.0:
                total_cols = i
                break

        # calculate first maxima of black points so as to get second point of the slope line
        y2, x2 = self.find_maxima(im_, row_start_idx, col_start_idx, col_start_idx + total_cols)

        # x1, y1, are the first point of the slope line and x2, y2 are the second points
        x1, y1 = 0, im_[:, pad_size].tolist().index(0)
        c = im_[:, pad_size].tolist().index(0)
        m = 0
        if y2 - y1 != 0:
            m = (x2 - x1) / (y2 - y1)
        else:
            process = False
            p_img = im_
        y = []

        # if a slope is detected then shift the pixels in the image to make it straight otherwise pass this step
        if process:
            for i in range(im_.shape[0]):
                y.append((((i * m) + c), i))
            processed = []
            for i in zip(im_, y):
                li = deque(i[0])
                count = int(i[1][0])
                if count > 0:
                    while count != 0:
                        li.popleft()
                        count -= 1
                    li = np.array(li)
                    li = np.lib.pad(li, (0, i[0].shape[0] - len(li)), 'maximum', )

                else:
                    count = abs(count)
                    while count != 0:
                        li.pop()
                        count -= 1
                    li = np.array(li)
                    li = np.lib.pad(li, (i[0].shape[0] - len(li), 0), 'maximum', )
                processed.append(li)

            p_img = np.array(processed)
        return p_img
