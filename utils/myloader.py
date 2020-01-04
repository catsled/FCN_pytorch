import cv2
import os
import numpy as np


class MyLoader(object):

    def __init__(self, path_src="", path_label="", transforms=None):
        self.path_src = path_src
        self.path_label = path_label
        self.transforms = transforms
        self.train_list = []
        self.label_list = []
        self._loadD()

    def __getitem__(self, index):
        l_offset = index * 32
        img_p, label_ps = self.train_list[index], self.label_list[l_offset:l_offset+32]
        img = cv2.imread(img_p)
        try:
            st = label_ps[0]
            label_ = cv2.imread(st)
            label_ = cv2.cvtColor(label_, cv2.COLOR_BGR2GRAY)
            label = label_
            for label_p in label_ps[1:]:
                label_ = cv2.imread(label_p)
                label_ = cv2.cvtColor(label_, cv2.COLOR_BGR2GRAY)
                label = np.dstack((label_, label))
            if self.transforms:
                img = self.transforms(img)
                label = self.transforms(label)
            return img, label
        except:
            pass

    def __len__(self):
        return len(self.train_list)

    def _loadD(self):
        i = 0
        for root, dirs, files in os.walk(self.path_src):
            for f in files:
                if f.endswith('.txt'):
                    continue
                abs_path = os.path.join(root, f)
                if not (f.endswith("_L.png")):
                    self.train_list.append(abs_path)

        for root, dirs, files in os.walk(self.path_label):
            for f in files:
                abs_path = os.path.join(root, f)
                self.label_list.append(abs_path)

        self.train_list = sorted(self.train_list, key=lambda x: x.split("_")[-1].split(".")[0])
        self.label_list = sorted(self.label_list, key=lambda x: (x.split("_")[1], int(x.split("_L")[-1].replace(".png", ""))))

