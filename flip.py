import random
import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def horizontalFlip_img(img):
    img = img[:, ::-1, :]
    return img

def horizontalFlip_bbox(img, bbox):
    img_center = np.array(img.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    bbox[:, [0, 2]] += 2 * (img_center[[0, 2]] - bbox[:, [0, 2]])

    box_w = abs(bbox[:, 0] - bbox[:, 2])

    bbox[:, 0] -= box_w
    bbox[:, 2] += box_w
    return bbox


def verticalFlip_img(img):
    img = img[::-1, :, :]
    return img


def verticalFlip_bbox(img, bbox):
    img_center = np.array(img.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))

    bbox[:, [1, 3]] += 2 * (img_center[[1, 3]] - bbox[:, [1, 3]])

    box_h = abs(bbox[:, 1] - bbox[:, 3])

    bbox[:, 1] -= box_h
    bbox[:, 3] += box_h

    return bbox


def draw_rect(im, cords, color=None):
    im = im.copy()
    if not color:
        color = [255, 255, 255]
    for cord in cords:
        pt1, pt2 = (float(cord[0]), float(cord[1])), (float(cord[2]), float(cord[3]))

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2]) / 200))
    return im


if __name__ == '__main__':
    csv_dir = ''
    img_dir = 'images\\'
    new_img_dir = 'imgs_h_v\\'
    new_csv_dir = ''

    classes = ['upper_right', 'upper_left', 'lower_left', 'lower_right']

    df = pd.read_csv(csv_dir + 'quads_data_v4.csv')
    data_number = int(len(df) / 4)
    data_idx = list(range(data_number))
    data_list = [df.iloc[i * 4]['file_name'] for i in data_idx]

    # new_df = pd.read_csv(csv_dir + 'flip_h_v.csv')
    new_df = pd.DataFrame(columns=['file_name', 'width', 'height', 'x_min', 'x_max', 'y_min', 'y_max', 'class_name', 'image_source'])

    ctr = 1
    for idx, row in df.iterrows():
        img_filename = os.path.join(img_dir, row["file_name"])
        print(img_filename)
        img = cv2.imread(img_filename)[:, :, ::-1]
        # if ctr == 1:
        #     # new_img_h = horizontalFlip_img(img)
        #
        #
        #     new_name = row["file_name"].split('.')[0]
        #     # new_img_name = new_name + '_h.jpg'
        #     # cv2.imwrite(os.path.join(new_img_dir, new_img_name), new_img_h)
        #
        #     new_img_v = verticalFlip_img(img)
        #     new_img_name = new_name + '_v.jpg'
        #     cv2.imwrite(os.path.join(new_img_dir, new_img_name), new_img_v)

        bbox = np.empty([1, 4])

        bbox[0, 0] = row["x_min"]
        bbox[0, 1] = row["y_min"]
        bbox[0, 2] = row["x_max"]
        bbox[0, 3] = row["y_max"]

        # print(bbox)
        new_name = row["file_name"].split('.')[0]
        new_bbox_h = horizontalFlip_bbox(img, bbox)
        new_bbox_v = verticalFlip_bbox(img, bbox)
        new_df = new_df.append({'file_name': new_name+ '_h.jpg' , 'width': row['width'], 'height': row['height'], 'x_min': new_bbox_h[0, 0], 'x_max': new_bbox_h[0, 2], 'y_min': new_bbox_h[0, 1],
                                'y_max': new_bbox_h[0, 3], 'class_name': row['class_name'], 'image_source': row['image_source']}, ignore_index = True)
        new_df = new_df.append({'file_name': new_name + '_v.jpg', 'width': row['width'], 'height': row['height'],
                                'x_min': new_bbox_v[0, 0], 'x_max': new_bbox_v[0, 2], 'y_min': new_bbox_v[0, 1],
                                'y_max': new_bbox_v[0, 3], 'class_name': row['class_name'],
                                'image_source': row['image_source']}, ignore_index=True)

        # print(new_bbox)
        ctr = ctr + 1
        if ctr == 5:
            print("new")
            ctr = 1

        # print(new_df)
        # plt.imshow(draw_rect(new_img_v, new_bbox))
        # plt.show()
        # break

    filepath = Path('flip_h_v.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(filepath, index=False)
