import cv2 as cv2
import numpy as np
import glob
import os

image_size = 16
# the big big loop
# for creating small pics of images
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 'kitchen',
         'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']

#for s in names:
#    image_list=[]
    # number of image
#    k = 0
#    for filename in glob.glob('training/'+s+'/*.jpg'):
#        img = cv2.imread(filename, 0).astype(float)
#        image_list.append(img)

        # crop image
        # if height > width => crop height
        # if weight > height => crop weight
#        height, width = img.shape
#        if height > width:
#            crop_img = img[0:width,0:width]
#        elif width > height:
#            crop_img = img[0:height,0:height]
#        else:
#            crop_img=img
        # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
#        resized_image = cv2.resize(crop_img, (image_size, image_size))
#        directory = 'training1/'+s+'/'+str(k)+'.jpg'
#        if not os.path.exists(directory):
#            os.makedirs(directory)
#        cv2.imwrite(directory ,resized_image)
#        k += 1
#        print ('saved '+s)

#
#
#cv2.imshow("cropped", resized_image/255)
#cv2.waitKey(0)

# now reading this images to a big-big vector
#

for s in names:
    image_list=[]
    k = 0
    for filename in glob.glob('training1/'+s+'/*.jpg'):
        img = cv2.imread(filename, 0).astype(float)
        # create a big vector from each rows of the image
        # the vector will be 16*16
        height, width = img.shape
        lst = []
        for h in range(height):
            lst.append(img[h])
        k+=1

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in xrange(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
