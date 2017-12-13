import numpy as np
import matplotlib as mp
import sklearn as sk
from sklearn.cluster import KMeans
import cv2
import glob
from PIL import Image
from sklearn import preprocessing
import collections
import glob
import os

image_size = 16
# the big big loop
# for creating small pics of images
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 'kitchen',
         'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']

for s in names:
    image_list=[]
   # number of image
    k = 0
    for filename in glob.glob('training/'+s+'/*.jpg'):
        img = preprocessing.normalize(cv2.imread(filename, 0).astype(float))
        image_list.append(img)

       # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
        directory = 'training2/'+s+'/'+str(k)

        k = 8
        k_k = k * k
        n = 4
        number_of_clusters = 8

        x_for_clusters = np.array([])

        # the big big loop
        # for filename in glob.glob('testing/*.jpg'):
        #    img1 = cv2.imread(filename, 0)
        #    image_list.append(img1)
#        img1 = preprocessing.normalize(cv2.imread('training/bedroom/0.jpg', 0).astype(float))
        height, width = img.shape

        # there must be a loop
        # for all the images in the folder
        y = 0
        while (y + k <= height):

            x = 0
            while (x + k <= width):
                image_patch = np.array([])
                patch = img[y:y + k, x:x + k]
                array_for_patch = np.array([])
                # image patch into the vector [8]*[8] = [64]
                for i in range(k):
                    for j in range(k):
                        array_for_patch = np.append(array_for_patch, patch[i][j])
                image_patch = np.append(image_patch, array_for_patch)
                # all patches is mean 0 and deviation is 1
                normalized_patch = preprocessing.scale(image_patch)
                normalized_patch = np.array(normalized_patch)
                if len(x_for_clusters) > 0:
                    x_for_clusters = np.vstack((x_for_clusters, normalized_patch))
                else:
                    x_for_clusters = np.append(x_for_clusters, normalized_patch)
                x += n

            y += n




        if not os.path.exists(directory):
            os.makedirs(directory)
        cv2.imwrite(directory ,resized_image)
        k += 1
        print ('saved '+s)
