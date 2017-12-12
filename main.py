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

x_for_clusters = np.array([])

# the big big loop
#for filename in glob.glob('testing/*.jpg'):
#    img1 = cv2.imread(filename, 0)
#    image_list.append(img1)
img1 = preprocessing.normalize(cv2.imread('training/bedroom/0.jpg', 0).astype(float))
height, width = img1.shape


# there must be a loop
# for all the images in the folder
y = 0
while (y + k<=height):

    x = 0
    while (x + k<=width):
        image_patch = np.array([])
        patch = img1[y:y+k, x:x+k]
        array_for_patch = np.array([])
        # image patch into the vector [8]*[8] = [64]
        for i in range(k):
            for j in range(k):
                array_for_patch = np.append(array_for_patch, patch[i][j])
        image_patch = np.append(image_patch, array_for_patch)
        # all patches is mean 0 and deviation is 1
        normalized_patch = preprocessing.scale(image_patch)
        normalized_patch = np.array(normalized_patch)
        if len(x_for_clusters)>0:
            x_for_clusters = np.vstack((x_for_clusters, normalized_patch))
        else:
            x_for_clusters = np.append(x_for_clusters, normalized_patch)
        x += n

    y += n

#
# K-Mean classifier!
#

# all the features in [0,1]
#normalized_x = preprocessing.normalize(x_for_clusters)
# all features is mean 0 and deviation is 1
#standardized_x = preprocessing.scale(x_for_clusters)

# number of clusters
kmeans = KMeans(number_of_clusters)
# fitting the input data
kmeans = kmeans.fit(x_for_clusters)

# centroid value
#centroids = kmeans.cluster_centers_

# normalize all the patches!!!!! before classification !!!!

# so we have 500 clusters
# now time for histograms

# loop for the all the patches in picture
# if this patches is in the cluster , then the number of this cluster + 1

histogram = np.zeros(number_of_clusters)
y = 0
while (y + k<=height):
    x = 0
    while (x + k<=width):
        image_patch = np.array([])

        patch = img1[y:y+k, x:x+k]
        array_for_patch = np.array([])
        for i in range(k):
            for j in range(k):
                array_for_patch = np.append(array_for_patch, patch[i][j])
        image_patch = np.append(image_patch, array_for_patch)
        # all patches is mean 0 and deviation is 1
        normalized_patch = preprocessing.scale(image_patch)
        label = kmeans.predict([normalized_patch])
        # all this labels go to represent an histogram
        histogram[label] += 1
        x += n

    y += n
print (histogram)