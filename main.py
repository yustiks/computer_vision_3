import numpy as np
import matplotlib as mp
import sklearn as sk
import cv2
import glob
from PIL import Image

k = 8
n = 4
number_of_clusters = 8
image_patches = []
vector_of_image = []
# the big big loop
#for filename in glob.glob('testing/*.jpg'):
#    img1 = cv2.imread(filename, 0)
#    image_list.append(img1)
img1 = cv2.imread('training/bedroom/0.jpg', 0).astype(float)
height, width = img1.shape
y = 0
while (y + n<=height):
    x = 0
    while (x + k<=width):
        patch = img1[y:y+k, x:x+k]
        image_patches.append(patch)
        x += n
    y += 4
vector_of_image.append(image_patches)

#
# K-Mean classifier!
#

# Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Set flags (Just to avoid line break in the code)
flags = cv2.KMEANS_RANDOM_CENTERS

# Apply KMeans
compactness,labels,centers = cv2.kmeans(vector_of_image,number_of_clusters,None,criteria,10,flags)

print (centers)