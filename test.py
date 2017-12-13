import numpy as np
import matplotlib.pyplot as plt
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
kmeans = 0
histograms = np.zeros([number_of_classes, number_of_clusters])

names1 = ['bedroom']
names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity', 'kitchen',
         'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store', 'Street', 'Suburb', 'TallBuilding']


input_user = 0
while(input_user!=8):
    print ('Choose the task: \n 1) create patches \n 2) download patches last created'
           '\n 3) train K-Mean classifier \n 4) download K-Mean classifier'
           '\n 5) create histograms for classes \n 6) download histograms for images '
           '\n 7) choose image for classification \n 8) exit \n')

    print('input number of task')
    input_user = int(input())
    if input_user == 1:
        # create patches
        print('1) creating patches....')
        number_images = 0
        for s in names:
            amount_patches = 0
            x_for_clusters = np.array([])
            k = 0
            for filename in glob.glob('training/' + s + '/*.jpg'):
                print('image' + str(filename))
                img1 = preprocessing.normalize(cv2.imread(filename, 0).astype(float))
                img1 = preprocessing.scale(img1)

                height, width = img1.shape

                y = 0
                while (y + size_patch <= height):
                    x = 0
                    while (x + size_patch <= width):
                        image_patch = np.array([])
                        amount_patches += 1
                        patch = img1[y:y + size_patch, x:x + size_patch]
                        x_for_clusters = np.append(x_for_clusters, patch)
                        x += n
                    y += n
                k += 1
                print('k = ' + str(k))
                number_images += 1
            x_for_clusters = x_for_clusters.reshape(amount_patches, size_patch * size_patch)
            file_output = open('data/' + s + '.pkl', 'wb')
            pickle.dump(x_for_clusters, file_output, -1)
            file_output.close()

        print('number of images = ' + str(number_images))
        print('done \n')

    elif input_user == 2:
        print('2) downloading patches last created')
        x_for_clusters = np.array([])
        patches_number = 0
        for s in names:
            file_input = open('data/' + s + '.pkl', 'r')
            data = pickle.load(file_input)
            patches_number += data.shape[0]
            if len(x_for_clusters) > 0:
                x_for_clusters = np.append(x_for_clusters, data)
            else:
                x_for_clusters = np.array(data)
            file_input.close()
        sh = data.shape
        x_for_clusters = x_for_clusters.reshape(patches_number, sh[1])
        print('done \n')

    elif input_user == 3:
        print('3) train K-means classifier')
        # number of clusters
        kmeans = KMeans(number_of_clusters)

        # fitting the input data
        kmeans = kmeans.fit(x_for_clusters)

        file_output = open('data/kmean.data', 'wb')
        pickle.dump(kmeans, file_output, -1)
        file_output.close()
        print('done \n')

    elif input_user == 4:
        print('4) download K-means classifier')
        file_input = open('data/kmean.data', 'r')
        kmeans = pickle.load(file_input)
        file_input.close()
        print('done \n')

    elif input_user == 5:
        print('5) create histograms for all the classes')
        if (kmeans==0):
            print ('kmeans haven`t been trained')
        else:
            histograms = np.zeros([number_of_classes, number_of_clusters])
            patches_number = 0
            k = 0
            for s in names:
                file_input = open('data/' + s + '.pkl', 'r')
                data = pickle.load(file_input)
                patches_number = data.shape[0]
                for i in range(patches_number):
                    label = kmeans.predict([data[i]])
                    # all this labels go to represent an histogram
                    histograms[k][label] += 1
                file_input.close()
                k += 1

        file_output = open('data/histogram.data', 'wb')
        pickle.dump(histograms, file_output, -1)
        file_output.close()
        print('done \n')


    elif input_user == 6:
        print('6) download histograms for images')
        file_input = open('data/histogram.data', 'r')
        histograms = pickle.load(file_input)
        file_input.close()
    elif input_user == 7:
        print('7) choose image for classification')
        filename = 'testing/'
        f = input('print the name of the file (number from 1 to 2987)')
        # so no we put image here and classify if it is from 1st of 15th class
        # the distance is calculated by cosine transform
        # filename = 'testing/33.jpg'
        filename = filename + str(f) + '.jpg'
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
                patch = patch.reshape(1, size_patch * size_patch)
                label = kmeans.predict(patch)
                histogram_for_input[label] += 1
                x += n
                print('label ' + str(label) + ' histogram[label] = ' + str(histogram_for_input[label]))
            y += n
        # cv2.imwrite('img_for_test.png', img)
        # cv2.imshow("outImg", img/255)
        # cv2.waitKey(0)
        plt.imshow(img/255, cmap='gray')
        plt.show()
        print('done \n')


    elif input_user == 8:
        print('8) exiting')
