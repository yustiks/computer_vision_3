# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:09:30 2017

@author: Dave
"""

import cv2
import numpy as np
import glob
#import os
import math
# debug: imageviewer from previous work 
#from CV1_convolution import show_image
from scipy import stats

# the big big loop
# for creating small pics of images
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 
         'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store',
         'Street', 'Suburb', 'TallBuilding']

# names as a dictionary to allow numeric labels
names_dict = {0:'bedroom', 1:'Coast', 2:'Forest', 3:'Highway', 4:'industrial', 
              5:'insidecity', 6:'kitchen', 7:'livingroom', 8:'Mountain', 
              9:'Office', 10:'OpenCountry', 11:'store', 12:'Street', 
              13:'Suburb', 14:'TallBuilding'}

labels_dict = {'bedroom':0, 'Coast':1, 'Forest':2, 'Highway':3, 'industrial':4, 
              'insidecity':5, 'kitchen':6, 'livingroom':7, 'Mountain':8, 
              'Office':9, 'OpenCountry':10, 'store':11, 'Street':12, 
              'Suburb':13, 'TallBuilding':14}
#
#tiny_img_dict={}
## The below code creates our initial image dictionary from the above dict
#for k in names_dict:
#    # number of image
#    for filename in glob.glob('training/'+tiny_img_dict[k]+'/*.jpg'):
#        #img_list = []
#        img = np.array(cv2.imread(filename, 0).astype(float))
#        img_list = [unitise(tiny_resize(crop(img))),k]
#        # We want to associate both the image itself and its labels with our
#        # identifier - in this case the image name
#        tiny_img_dict[filename] = img_list
#        #image_list.append(img)
        
# below unpacks the above array and labels into a 17xt array
        
#
img_dict = dict()
tiny_img_dict = dict()
img_list = []
tiny_img_list = []

# The below code creates our initial image dictionary
for s in names:
    # number of image
    k = 0
    for filename in glob.glob('training/'+s+'/*.jpg'):
        #img_list = []
        img = np.array(cv2.imread(filename, 0).astype(float))
        img_list = [img,s]
        # We want to associate both the image itself and its labels with our
        # identifier - in this case the image name
        img_dict[filename] = img_list
        #image_list.append(img)
        
#probably a better way to do this, but for now just re-looping through to 
        # make dict of tiny images
for s in names:
    # number of image
    k = 0
    for filename in glob.glob('training/'+s+'/*.jpg'):
        #img_list = []
        img = np.array(cv2.imread(filename, 0).astype(float))
        img_list = [unitise(tiny_resize(crop(img))),s]
        # We want to associate both the image itself and its labels with our
        # identifier - in this case the image name
        tiny_img_dict[filename] = img_list
#

#
##tiny_img_dict = 
## We then need to crop the images in the dictionary - can access by using list
#        # and indexing - remember first index is number in dictionary, 2nd
#        # index is [0] for image, [1] for label

def crop(img):
    """A function to resize an ixj image to ixi or jxj, depending
    on the smaller value. Crops borders of larger dimension"""
    # We're applying this to all values, so passes if type is a string
    # Might need to make more robust by checking if type is ndarray or image
    if isinstance(img, __builtins__.str):
        return img
    else: 
        height, width = img.shape
        if height > width:
            border = math.floor((height-width)/2)
            crop_img = img[border:width+border, 0:width]
        elif width > height:
            border = math.floor((width-height)/2)
            crop_img = img[0:height, border:height+border]
        else:
            crop_img=img
        return crop_img
        
# We then need to resize the images in the dictionary
def tiny_resize(img, img_size = 16):
    """Takes a square image and resizes to ixi, default value being 16x16"""
    # Currently just using np resize function
    if isinstance(img, __builtins__.str):
        return img
    else:
        return np.resize(img, (img_size,img_size))
    
def unitise(img):
    """Takes an ndarray, returns a normalised ndarray of the same size with
    mean of 0, Euclidean length of 1"""
    if isinstance(img, __builtins__.str):
        return img
    else:
        unit_img = img - (np.mean(img) * np.ones(img.shape))
        if np.linalg.norm(unit_img) == 0:
            pass
        else:
            unit_img = unit_img / np.linalg.norm(unit_img)
        return unit_img

# Unpack tiny_img_dict into form for Nearest Neighbour
knn_input = np.zeros((1500,256), dtype=np.float)
knn_target = np.zeros((1500,1), dtype=np.uint)
for i in range(len(tiny_img_dict)):
    knn_input[i] = np.reshape(list(tiny_img_dict.values())[i][0],(1,256))
    knn_target[i] = labels_dict[list(tiny_img_dict.values())[i][1]]


class NearestNeighbour(object):
  def __init__(self, starting_k):
    self.starting_k = starting_k

  def train(self, data, labels):
    """Assume X is an ndarray, Y is 1-dimension of size N """
    # the nearest neighbour classifier simply remembers all the training data
    self.train_data = data
    self.train_labels = labels

  def test(self, data):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test_cases = data.shape[0]
    # lets make sure that the output type matches the input type
    predicted_labels = np.zeros(num_test_cases, dtype = np.uint)

    # loop over all test rows
    for i in range(num_test_cases):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      train_distance = np.sum(np.abs(self.train_data - data[i,:]), axis=1)
      #print('Distances', train_distance)
      min_indices = np.argsort(train_distance)[:self.starting_k]
      #print('Indices', min_indices)
      #print('Labels', predicted_labels)
      #print('Predicted Label i OG', predicted_labels[i])
      #print('Self Train Labels', self.train_labels[min_indices])
      #print('Stats mode', stats.mode(self.train_labels[min_indices])[0])
      predicted_labels[i] = stats.mode(self.train_labels[min_indices])[0]
      #print('Predicted Label i AA', predicted_labels[i])

    return predicted_labels