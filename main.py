import numpy as np
import matplotlib as mp
import sklearn as sk
from sklearn.cluster import KMeans
import cv2
import glob
from PIL import Image
from sklearn import preprocessing
import collections
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial

size_patch = 100

n = 100
number_of_clusters = 50
number_of_classes = 15

names1 = ['bedroom']
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 'kitchen',
         'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']

'''
number_images = 0
for s in names:
    amount_patches = 0
    x_for_clusters = np.array([])
    k = 0
    for filename in glob.glob('training/'+s+'/*.jpg'):
        print('image'+str(filename))
        img1 = preprocessing.normalize(cv2.imread(filename, 0).astype(float))
        img1 = preprocessing.scale(img1)

        height, width = img1.shape

        y = 0
        while (y + size_patch<=height):
            x = 0
            while (x + size_patch<=width):
                image_patch = np.array([])
                amount_patches += 1
                patch = img1[y:y+size_patch, x:x+size_patch]
                x_for_clusters = np.append(x_for_clusters, patch)
                x += n
            y += n
        k+=1
        print('k = '+ str(k))
        number_images += 1
    x_for_clusters = x_for_clusters.reshape(amount_patches, size_patch * size_patch)
    output = open('data/'+s+'.pkl', 'wb')
    pickle.dump(x_for_clusters, output, -1)
    output.close()
    

print('number of images = '+str(number_images))
'''

#
# K-Mean classifier!
#
'''
x_for_clusters = np.array([])
patches_number = 0
for s in names:
    input = open('data/'+s+'.pkl', 'r')
    data = pickle.load(input)
    patches_number += data.shape[0]
    if len(x_for_clusters)>0:
        x_for_clusters = np.append(x_for_clusters,data)
    else:
        x_for_clusters = np.array(data)
    input.close()
sh = data.shape
x_for_clusters = x_for_clusters.reshape(patches_number,sh[1])
print ('done')


# number of clusters
kmeans = KMeans(number_of_clusters)

# fitting the input data
kmeans = kmeans.fit(x_for_clusters)

output = open('data/kmean.data', 'wb')
pickle.dump(kmeans, output, -1)
output.close()
'''
# so we have 500 clusters
# now time for histograms
input = open('data/kmean.data', 'r')
kmeans = pickle.load(input)
input.close()
# loop for the all the patches in picture
# if this patches is in the cluster , then the number of this cluster + 1
'''
histograms = np.zeros([number_of_classes,number_of_clusters])
patches_number = 0
k = 0
for s in names:
    input = open('data/'+s+'.pkl', 'r')
    data = pickle.load(input)
    patches_number = data.shape[0]
    for i in range(patches_number):
        label = kmeans.predict([data[i]])
    # all this labels go to represent an histogram
        histograms[k][label] += 1
    input.close()
    k += 1

output = open('data/histogram.data', 'wb')
pickle.dump(histograms, output, -1)
output.close()
'''
input = open('data/histogram.data', 'r')
histograms = pickle.load(input)
input.close()
# so no we put image here and classify if it is from 1st of 15th class
# the distance is calculated by cosine transform
filename = 'testing/33.jpg'
img = preprocessing.normalize(cv2.imread(filename, 0).astype(float))
img = preprocessing.scale(img)

# making patches for 1 input image
height, width = img.shape
y = 0
amount_patches = 0
histogram_for_input = np.zeros([number_of_clusters])
while (y + size_patch <= height):
    x = 0
    while (x + size_patch <= width):
        image_patch = np.array([])
        amount_patches += 1
        patch = img[y:y + size_patch, x:x + size_patch]
        patch = patch.reshape(1, size_patch*size_patch)
        label = kmeans.predict(patch)
        histogram_for_input[label] += 1
        x += n
        print('label ' + str(label)+ ' histogram[label] = '+str(histogram_for_input[label]))
    y += n

# compare out_instagram with instagram from other clusters
max = 2
cluster_output = 0
for i in range(number_of_classes):
    t1 = histogram_for_input;
    t2 = histograms[i];
    result = 1 - spatial.distance.cosine(t1, t2)
    if max ==2:
        #max = cosine_similarity(t1,t2)
        max = result
        cluster_output = i
    elif (max<result):
        max = result
        cluster_output = i
print ('class number ', cluster_output, ' name ', names[cluster_output])
print ('done')



