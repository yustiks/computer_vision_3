# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:08:31 2017

@author: Dave
"""

import cv2
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from CV3_1_functions import *
#import os
# debug: imageviewer from previous work 
#from CV1_convolution import show_image


# given labels for our images
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 
         'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store',
         'Street', 'Suburb', 'TallBuilding']

# dictionaries mapping names to labels and vice versa - used so we can 
# consistently use ndarrays, while preserving the ability to output text labels
names_dict = {0:'bedroom', 1:'Coast', 2:'Forest', 3:'Highway', 4:'industrial', 
              5:'insidecity', 6:'kitchen', 7:'livingroom', 8:'Mountain', 
              9:'Office', 10:'OpenCountry', 11:'store', 12:'Street', 
              13:'Suburb', 14:'TallBuilding'}

labels_dict = {'bedroom':0, 'Coast':1, 'Forest':2, 'Highway':3, 'industrial':4, 
              'insidecity':5, 'kitchen':6, 'livingroom':7, 'Mountain':8, 
              'Office':9, 'OpenCountry':10, 'store':11, 'Street':12, 
              'Suburb':13, 'TallBuilding':14}
        
img_dict = dict()
tiny_img_dict = dict()

# The below code creates our initial image dictionary
for s in names:
    for filename in glob.glob('training/'+s+'/*.jpg'):
        img_list = []
        img = np.array(cv2.imread(filename, 0).astype(float),dtype=np.float)
        img_list = [img,s]
        # We want to associate both the image itself and its label with our
        # identifier - in this case the image name
        img_dict[filename] = img_list
        
#probably a better way to do this, but for now just re-looping through to 
        # make dict of tiny images
for s in names:
    for filename in glob.glob('training/'+s+'/*.jpg'):
        img_list = []
        img = np.array(cv2.imread(filename, 0).astype(float),dtype=np.float)
        img_list = [unitise(tiny_resize(crop(img))),s]
        # We want to associate both the image itself and its labels with our
        # identifier - in this case the image name
        tiny_img_dict[filename] = img_list


# Unpack tiny_img_dict into form for Nearest Neighbour
knn_input = np.zeros((1500,256), dtype=np.float)
knn_target = np.zeros((1500,1), dtype=np.uint)
for i in range(len(tiny_img_dict)):
    knn_input[i] = np.reshape(list(tiny_img_dict.values())[i][0],(1,256))
    knn_target[i] = labels_dict[list(tiny_img_dict.values())[i][1]]
    
# Perform validation for a range of ks and using several random seeds
#val_matrix = np.zeros((100,5))
#seed_num = 0
#for seed in [23, 37, 42, 88, 101]:
#    for k in range(100):
#        val_matrix[k][seed_num] = validate_results(k+1, knn_input, 
#                  knn_target, seed)[0] 
#    seed_num += 1
    
# np.max(np.mean(val_matrix, axis=1)) comes at k = 24. 


# Finally, import the test images, run our best classifier on them and output
test_img_dict = dict()
for filename in glob.glob('testing/*.jpg'):
        img_list = []
        img = np.array(cv2.imread(filename, 0).astype(float),dtype=np.float)
        img_list = [unitise(tiny_resize(crop(img)))]        
        # We want to associate both the image itself and its label with our
        # identifier - in this case the image name
        test_img_dict[filename] = img_list
        
knn_test_input = np.zeros((2985,256), dtype=np.float)
#knn_test_filename = np.empty((2985,1), dtype=np.string_)
knn_test_filename = []

for i in range(len(test_img_dict)):
    knn_test_filename.append(list(test_img_dict.keys())[i][8:])
    knn_test_input[i] = np.reshape(list(test_img_dict.values())[i][0],(1,256))
    
TestKNN = KNN_Classifier(24)
TestKNN.train(knn_input, knn_target)
results = TestKNN.test(knn_test_input)

label_results = []
for i in range(len(results)):
    label_results.append(names_dict[results[i][0]])
    
output = ''
for i in range(len(label_results)):
    output += str(knn_test_filename[i])
    output += ' '
    output += str(label_results[i])
    output += '\n'
f = open('run1.txt','w')
f.write(output)
f.close()