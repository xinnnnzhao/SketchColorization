import cv2
import numpy as np
import os


def read_img(size):
    # Read in images, return 2 numpy array
    original_path = "../../resource/training_data_set/colored_data_set"
    lineart_path = "../../resource/training_data_set/sketch_data_set"

    original_files = sorted(os.listdir(original_path))
    lineart_files = sorted(os.listdir(lineart_path))

    X_o = []
    X_l = []

    for i, img in enumerate(lineart_files):
        if i < size:
            img_lineart_path = os.path.join(lineart_path, img)
            img_original_path = os.path.join(original_path, img)

            img_lineart = cv2.imread(img_lineart_path)
            img_original = cv2.imread(img_original_path)

            img_original = cv2.resize(img_original, (512, 512))
            img_lineart = cv2.resize(img_lineart, (512, 512))
            img_lineart = cv2.cvtColor(img_lineart, cv2.COLOR_RGB2GRAY)
            img_lineart = np.expand_dims(img_lineart, 3)
            X_o.append(img_original)
            X_l.append(img_lineart)
        else:
            break

    X_o = np.array(X_o)
    X_l = np.array(X_l)
    return X_o, X_l
