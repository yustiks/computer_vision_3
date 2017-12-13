# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:08:14 2017

@author: Dave
"""

import math
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

class KNN_Classifier(object):
    
    def __init__(self, starting_k):
        """Initialises a KNN_Classifier, taking argument k - 
        the number of neighbours used for comparison"""
        self.starting_k = starting_k

    def train(self, data, labels):
        """Data should be an ndarray of floats, labels an ndarray of uints. 
        Train simply stores these values for use in the test function"""
        self.train_data = data
        self.train_labels = labels

    def test(self, data):
        """Takes in an ndarray of same shape as training data, outputs an 
        ndarray of labels"""
        num_test_cases = data.shape[0]
        # initialises label output - uses int labels to retain precision
        predicted_labels = np.zeros((num_test_cases,1), dtype = np.uint)

        # For every test case, calculates absolute distance to each row of the
        # training data, takes indices of k values with least distance, then
        # sets the predicted label to be the mode of these values
        for i in range(num_test_cases):
          train_distance = np.sum(np.abs(self.train_data - data[i,:]), axis=1)
          min_indices = np.argsort(train_distance)[:self.starting_k]
          predicted_labels[i] = stats.mode(self.train_labels[min_indices])[0]

        return predicted_labels
    
def crop(img):
    """A function to resize an ixj image to ixi or jxj, depending
    on the smaller value. Crops borders of larger dimension"""
    # We're applying this to all values, so passes if type is a string
    # Might need to make more robust by checking if type is ndarray or image
#    if isinstance(img, __builtins__.str):
#        return img
#    else: 
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
#    if isinstance(img, __builtins__.str):
#        return img
#    else:
    return np.resize(img, (img_size,img_size))
    

def unitise(img):
    """Takes an ndarray, returns a normalised ndarray of the same size with
    mean of 0, Euclidean length of 1"""
#    if isinstance(img, __builtins__.str):
#        return img
#    else:
    unit_img = img - (np.mean(img) * np.ones(img.shape))
    if np.linalg.norm(unit_img) == 0:
        pass
    else:
        unit_img = unit_img / np.linalg.norm(unit_img)
    return unit_img


def validate_results(k, input_data, target_label, rand_seed=37):
    """Given a labelled training set of inputs and targets and a hyperparameter
    performs validation and outputs accuracy"""
    X_train,X_test,y_train,y_test = train_test_split(input_data,target_label,
                                        test_size=0.2,random_state=rand_seed) 
    KNN = KNN_Classifier(k)
    KNN.train(X_train, y_train)
    results = KNN.test(X_test)
    accuracy = sum(y_test == results)/300
    #accuracy = sum(np.reshape(y_test,(300,)) == results)/300
    return accuracy