import numpy as np
import matplotlib as mp
import sklearn as sk
from sklearn.cluster import KMeans
import cv2
import glob
from PIL import Image
from sklearn import preprocessing


k = 8
n = 4
number_of_clusters = 8

x_linear_regression = []
y_linear_regression = []
# the big big loop
#for filename in glob.glob('testing/*.jpg'):
#    img1 = cv2.imread(filename, 0)
#    image_list.append(img1)
img1 = cv2.imread('training/bedroom/0.jpg', 0).astype(float)
height, width = img1.shape
y = 0
while (y + k<=height):
    image_patches = np.array([])
    x = 0
    while (x + k<=width):
        patch = img1[y:y+k, x:x+k]
        array_for_patch = np.array([])
        for i in range(k):
            for j in range(k):
                array_for_patch = np.append(array_for_patch, patch[i][j])
        image_patches = np.append(image_patches, array_for_patch)
        x += n
    y += n
    y_linear_regression.append(1)
    x_linear_regression.append(image_patches)


#
# K-Mean classifier!
#

# all the features in [0,1]
normalized_x = preprocessing.normalize(x_linear_regression)
# all features is mean 0 and deviation is 1
standardized_x = preprocessing.scale(x_linear_regression)

kmeans = KMeans(number_of_clusters)
kmeans = kmeans.fit(standardized_x)
labels = kmeans.predict(standardized_x)
centroids =kmeans.cluster_centers_

print(centroids)