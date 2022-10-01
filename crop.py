import random
import numpy as np
import cv2
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


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


if __name__ == '__main__':
    csv_dir = ''
    img_dir = 'images\\'
    new_img_dir = 'crop_imgs\\'
    new_csv_dir = ''

    classes = ['upper_right', 'upper_left', 'lower_left', 'lower_right']
    ur_numbers = [18, 17, 16, 15, 14, 13, 12, 11]
    ul_numbers = [21, 22, 23, 24, 25, 26, 27, 28]
    lr_numbers = [48, 47, 46, 45, 44, 43, 42, 41]
    ll_numbers = [31, 32, 33, 34, 35, 36, 37, 38]

    quads_df = pd.read_csv(csv_dir + 'quadrant_data_edited.csv')
    numbering_df = pd.read_csv(csv_dir + 'numbering_data_edited.csv')
    data_number = int(len(quads_df) / 4)
    data_idx = list(range(data_number))
    data_list = [quads_df.iloc[i * 4]['file_name'] for i in data_idx]

    new_df = pd.DataFrame(
        columns=['file_name', 'width', 'height', 'x_min', 'x_max', 'y_min', 'y_max', 'class_name', 'image_source'])

    file_names = quads_df['file_name'].unique()

    for file in file_names:
        print(file)
        file_quads_df = quads_df.loc[(quads_df['file_name'] == file)]
        file_numbering_df = numbering_df.loc[(numbering_df['file_name'] == file)]

        img_filename = os.path.join(img_dir, file)
        img = cv2.imread(img_filename)[:, :, ::-1]

        new_name = file.split('.')[0]

        # upper right
        upper_right = file_quads_df.loc[(file_quads_df['class_name'] == 1)]
        ur_x_min = int(upper_right['x_min'])
        ur_x_max = int(upper_right['x_max'])
        ur_y_max = int(upper_right['y_max'])
        ur_y_min = int(upper_right['y_min'])
        cropped_image_ur = img[ur_y_min:ur_y_max, ur_x_min: ur_x_max]

        # plt.imshow(cropped_image_ur)
        # print(new_width)
        # print(new_height)
        # plt.show()

        file_name_ur = new_name + '_ur.jpg'
        cv2.imwrite(os.path.join(new_img_dir, file_name_ur), cropped_image_ur)

        new_width_ur = cropped_image_ur.shape[1]
        new_height_ur = cropped_image_ur.shape[0]

        # cords = []
        for number in ur_numbers:
            row = file_numbering_df.loc[(file_numbering_df['class_name'] == number)]

            if not row.empty:
                number_x_min = int(row['x_min'].values[0]) - ur_x_min
                number_x_max = int(row['x_max'].values[0]) - ur_x_min
                number_y_min = int(row['y_min'].values[0]) - ur_y_min
                number_y_max = int(row['y_max'].values[0]) - ur_y_min
                new_df = new_df.append(
                    {'file_name': file_name_ur, 'width': new_width_ur, 'height': new_height_ur,
                     'x_min': number_x_min, 'x_max': number_x_max, 'y_min': number_y_min,
                     'y_max': number_y_max, 'class_name': number,
                     'image_source': row['image_source'].values[0]}, ignore_index=True)
                # bbox = []
                #
                # bbox.append(number_x_min)
                # bbox.append(number_y_min)
                # bbox.append(number_x_max)
                # bbox.append(number_y_max)
                # cords.append(bbox)

        # plt.imshow(draw_rect(cropped_image_ur, cords))
        # plt.show()

        # upper left
        upper_left = file_quads_df.loc[(file_quads_df['class_name'] == 2)]
        ul_x_min = int(upper_left['x_min'])
        ul_x_max = int(upper_left['x_max'])
        ul_y_max = int(upper_left['y_max'])
        ul_y_min = int(upper_left['y_min'])
        cropped_image_ul = img[ul_y_min:ul_y_max, ul_x_min: ul_x_max]

        file_name_ul = new_name + '_ul.jpg'
        # flip
        flip_cropped_image_ul = horizontalFlip_img(cropped_image_ul)
        cv2.imwrite(os.path.join(new_img_dir, file_name_ul), flip_cropped_image_ul)

        new_width_ul = cropped_image_ul.shape[1]
        new_height_ul = cropped_image_ul.shape[0]

        # cords = []
        for number in ul_numbers:
            row = file_numbering_df.loc[(file_numbering_df['class_name'] == number)]

            if not row.empty:
                number_x_min = int(row['x_min'].values[0]) - ul_x_min
                number_x_max = int(row['x_max'].values[0]) - ul_x_min
                number_y_min = int(row['y_min'].values[0]) - ul_y_min
                number_y_max = int(row['y_max'].values[0]) - ul_y_min

                bbox = np.empty([1, 4])

                bbox[0, 0] = number_x_min
                bbox[0, 1] = number_y_min
                bbox[0, 2] = number_x_max
                bbox[0, 3] = number_y_max

                new_bbox = horizontalFlip_bbox(cropped_image_ul, bbox)

                new_df = new_df.append(
                    {'file_name': file_name_ul, 'width': new_width_ul, 'height': new_height_ul,
                     'x_min': new_bbox[0, 0], 'x_max': new_bbox[0, 2], 'y_min': new_bbox[0, 1],
                     'y_max': new_bbox[0, 3], 'class_name': number,
                     'image_source': row['image_source'].values[0]}, ignore_index=True)

                # cords.append(new_bbox)
                # print(bbox)

        # plt.imshow(draw_rect(flip_cropped_image_ul, cords))
        # plt.show()

        # lower left
        lower_left = file_quads_df.loc[(file_quads_df['class_name'] == 3)]
        ll_x_min = int(lower_left['x_min'])
        ll_x_max = int(lower_left['x_max'])
        ll_y_max = int(lower_left['y_max'])
        ll_y_min = int(lower_left['y_min'])
        cropped_image_ll = img[ll_y_min:ll_y_max, ll_x_min: ll_x_max]
        file_name_ll = new_name + '_ll.jpg'
        # flip
        flip_cropped_image_ll = horizontalFlip_img(cropped_image_ll)
        cv2.imwrite(os.path.join(new_img_dir, file_name_ll), flip_cropped_image_ll)

        new_width_ll = cropped_image_ll.shape[1]
        new_height_ll = cropped_image_ll.shape[0]

        # cords = []
        for number in ll_numbers:
            row = file_numbering_df.loc[(file_numbering_df['class_name'] == number)]

            if not row.empty:
                number_x_min = int(row['x_min'].values[0]) - ll_x_min
                number_x_max = int(row['x_max'].values[0]) - ll_x_min
                number_y_min = int(row['y_min'].values[0]) - ll_y_min
                number_y_max = int(row['y_max'].values[0]) - ll_y_min

                bbox = np.empty([1, 4])

                bbox[0, 0] = number_x_min
                bbox[0, 1] = number_y_min
                bbox[0, 2] = number_x_max
                bbox[0, 3] = number_y_max

                new_bbox = horizontalFlip_bbox(cropped_image_ll, bbox)

                new_df = new_df.append(
                    {'file_name': file_name_ll, 'width': new_width_ll, 'height': new_height_ll,
                     'x_min': new_bbox[0, 0], 'x_max': new_bbox[0, 2], 'y_min': new_bbox[0, 1],
                     'y_max': new_bbox[0, 3], 'class_name': number,
                     'image_source': row['image_source'].values[0]}, ignore_index=True)

                # cords.append(new_bbox)
                # print(bbox)

        # plt.imshow(draw_rect(flip_cropped_image_ll, cords))
        # plt.show()

        # lower right
        lower_right = file_quads_df.loc[(file_quads_df['class_name'] == 4)]
        lr_x_min = int(lower_right['x_min'])
        lr_x_max = int(lower_right['x_max'])
        lr_y_max = int(lower_right['y_max'])
        lr_y_min = int(lower_right['y_min'])
        cropped_image_lr = img[lr_y_min:lr_y_max, lr_x_min: lr_x_max]
        file_name_lr = new_name + '_lr.jpg'
        cv2.imwrite(os.path.join(new_img_dir, file_name_lr), cropped_image_lr)

        new_width_lr = cropped_image_lr.shape[1]
        new_height_lr = cropped_image_lr.shape[0]

        cords = []
        for number in lr_numbers:
            row = file_numbering_df.loc[(file_numbering_df['class_name'] == number)]

            if not row.empty:
                number_x_min = int(row['x_min'].values[0]) - lr_x_min
                number_x_max = int(row['x_max'].values[0]) - lr_x_min
                number_y_min = int(row['y_min'].values[0]) - lr_y_min
                number_y_max = int(row['y_max'].values[0]) - lr_y_min
                new_df = new_df.append(
                    {'file_name': file_name_lr, 'width': new_width_lr, 'height': new_height_lr,
                     'x_min': number_x_min, 'x_max': number_x_max, 'y_min': number_y_min,
                     'y_max': number_y_max, 'class_name': number,
                     'image_source': row['image_source'].values[0]}, ignore_index=True)
                bbox = []

                bbox.append(number_x_min)
                bbox.append(number_y_min)
                bbox.append(number_x_max)
                bbox.append(number_y_max)
                cords.append(bbox)

        # plt.imshow(draw_rect(cropped_image_lr, cords))
        # plt.show()
        # break

    filepath = Path('new_df.csv')
    filepath.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(filepath, index=False)
