# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:27:21 2017

Code originally taken from: https://raw.githubusercontent.com/tensorflow/
tensorflow/r1.4/tensorflow/examples/tutorials/layers/cnn_mnist.py

Original article: https://www.tensorflow.org/tutorials/layers

With modifications to fit data set by David Guest
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn.model_selection import train_test_split
from PIL import Image
#from CV3_1_functions import *

import numpy as np
import tensorflow as tf
#import cv2
import glob
import os
import math

tf.logging.set_verbosity(tf.logging.INFO)


# given labels for our images
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 
         'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store',
         'Street', 'Suburb', 'TallBuilding']

# dictionaries mapping names to labels and vice versa - used so we can eval_input
# consistently use ndarrays, while preserving the ability to output text labels
names_dict = {0:'bedroom', 1:'Coast', 2:'Forest', 3:'Highway', 4:'industrial', 
              5:'insidecity', 6:'kitchen', 7:'livingroom', 8:'Mountain', 
              9:'Office', 10:'OpenCountry', 11:'store', 12:'Street', 
              13:'Suburb', 14:'TallBuilding'}

labels_dict = {'bedroom':0, 'Coast':1, 'Forest':2, 'Highway':3, 'industrial':4, 
              'insidecity':5, 'kitchen':6, 'livingroom':7, 'Mountain':8, 
              'Office':9, 'OpenCountry':10, 'store':11, 'Street':12, 
              'Suburb':13, 'TallBuilding':14}

FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))+'/'
FILTER_MULT = 1

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

# Rewritten original import function to put training images and labels directly
# into ndarray        
img_arr_64_pix = np.zeros((1500, 4097), dtype=np.float32)
row_no = 0
for s in names:
    #print(s)
    #print(glob.glob('C:/Users/dg4n17/training/' + s + '/*.jpg'))
    for filename in glob.glob('C:/Users/dg4n17/training/' + s + '/*.jpg'):
        #print(filename)
        img = np.array(Image.open(filename),dtype=np.float32)
        img_64 = tiny_resize(crop(img), img_size = 64)
        img_arr_64_pix[row_no, 0:4096] = np.reshape(img_64, (1,4096))
        img_arr_64_pix[row_no, 4096] = labels_dict[s]
        row_no += 1


def cnn_model_fn_simplenet(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 64, 64, 1])

  # Convolutional Layer #1
  # Computes 32 features using a 5x5 filter with ReLU activation.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 64, 64, 1]
  # Output Tensor Shape: [batch_size, 64, 64, 32]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16*FILTER_MULT,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  
  conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=32*FILTER_MULT,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 64, 64, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 32, 32, 32]
  # Output Tensor Shape: [batch_size, 32, 32, 64]
  conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=32*FILTER_MULT,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 32, 32, 64]
  # Output Tensor Shape: [batch_size, 16, 16, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 16, 16, 64]
  # Output Tensor Shape: [batch_size, 16 * 16 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 16 * 16 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 16 * 16 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  # skip the dense later for this network
  #dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=pool2_flat, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 15]
  logits = tf.layers.dense(inputs=dropout, units=15)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=15)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  train_data,eval_data,train_labels,eval_labels = train_test_split(
          img_arr_64_pix[:,0:4096],img_arr_64_pix[:,4096],test_size=0.2,
          random_state=37)

  # Create the Estimator
  scene_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn_simplenet, model_dir="/tmp/scene_convnet_model_simplenet")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  scene_classifier.train(
      input_fn=train_input_fn,
      steps=1000, # steps reduced from 20k to 1k, to increase when satisfied with approach.
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = scene_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
