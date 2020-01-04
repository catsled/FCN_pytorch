import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import copy


if __name__ == '__main__':

    path = "../data/camseq01/label_colors.txt"
    color_list = []

    with open(path, 'r') as f:
        for l in f.readlines():
            l = l.strip().replace('\t', ' ').split(' ')
            r, g, b, k = l[0], l[1], l[2], l[3]
            color_list.append([int(b), int(g), int(r)])

    img_list = []
    for root, dirs, files in os.walk("../data/camseq01"):
        for f in files:
            if f.endswith('txt'):
                continue
            if not f.endswith("_L.png"):
                continue
            abs_path = os.path.join(root, f)
            img_list.append(abs_path)

    dest_dir = "../data/mask/"
    n_img = None
    n_mask = "{}"
    for f in img_list:
        img = cv2.imread(f)
        base_name = f.split("/")[-1].replace('.png', '')
        h, w, c = img.shape
        for n in range(len(color_list)):
            color = np.array(color_list[n], dtype=np.uint8)
            mask = copy.deepcopy(img)
            for i in range(h):
                for j in range(w):
                    if not ((mask[i, j, :] == color).all()):
                        mask[i, j, :] = np.array([0, 0, 0], dtype=np.uint8)
                    else:
                        mask[i, j, :] = np.array([255, 255, 255], dtype=np.uint8)
            n_p = dest_dir + base_name + n_mask.format(n) + '.png'
            cv2.imwrite(n_p, mask)




