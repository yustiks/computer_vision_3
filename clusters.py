import numpy as np
import matplotlib as mp
import sklearn as sk
from sklearn.cluster import KMeans
import cv2
import glob
from PIL import Image
from sklearn import preprocessing
import collections

k = 8
k_k = k*k
n = 4
number_of_clusters = 8

img1 = preprocessing.normalize(cv2.imread('training/bedroom/0.jpg', 0).astype(float))
img1 = preprocessing.scale(img1)
height, width = img1.shape

# big loop
y = 0
while (y + k<=height):
    x = 0
    while (x + k<=width):
        patch = img1[y:y+k, x:x+k]
        # all patches is mean 0 and deviation is 1

        if len(x_for_clusters)>0:
            x_for_clusters = np.vstack((x_for_clusters, patch))
        else:
            x_for_clusters = np.append(x_for_clusters, patch)
        x += n

    y += n
