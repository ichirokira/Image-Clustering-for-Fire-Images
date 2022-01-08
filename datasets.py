import os
from imutils import paths
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import cv2


def load_khi(data_path="data/khi_images"):
    image_paths = list(paths.list_images(data_path))
    print("[INFO]--------Load dataset-------------")
    y_names = {0: "keep", 1: "change"}
    x = []
    y = []
    for i, path in enumerate(image_paths):
        if i % 100 == 0:
            print("[INFO] Processing {}/{}".format(i, len(image_paths)))

        label = path.split("/")[-2]
        image = cv2.imread(path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x.append(gray)
        if label == "change":
            y.append(1)
        elif label == "keep":
            y.append(0)
    x = np.stack(x, axis=0)
    y = np.array(y)
    x = x.reshape((x.shape[0], -1))
    x = np.divide(x, 255.)
    return x, y, y_names



