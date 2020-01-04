import numpy as np
import matplotlib.pyplot as plt


def func_rearrangeRGB(img_batch):
    """
    This function aims to rearrange the img for viewing.
    :param img_batch: numpy [C X H X W]
    C: channels of image
    H: height of image
    W: width of image
    img_batch[0]: red;
    img_batch[1]: green;
    img_batch[2]: blue
    :return: img_batch can be show [H x W X C]
    """
    c, h, w = img_batch.shape
    r_im = img_batch[0]
    g_im = img_batch[1]
    b_im = img_batch[2]

    img_batch = np.dstack((r_im, g_im, b_im))

    return img_batch.reshape((h, w, c))


def func_showImage(img_batch):
    """
    This function aims to show image by matplotlib, for convenient
    you should preprocess the img_batch by torchvision.utils.make_grid
    :param img_batch:
    :return:
    """
    plt.imshow(img_batch)
    plt.show()


