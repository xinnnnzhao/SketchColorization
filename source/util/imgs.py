import cv2
import numpy as np
from random import randint


def imageblur(cimg, sampling=False):
    if sampling:
        cimg = cimg * 0.3 + np.ones_like(cimg) * 0.7 * 255
    else:
        for i in range(15):
            randx = randint(0, 205)
            randy = randint(0, 205)
            cimg[randx:randx + 50, randy:randy + 50] = 255
    return cv2.blur(cimg, (100, 100))
