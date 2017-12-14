import cv2
import numpy as np
import glob
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# import os
# debug: imageviewer from previous work
# from CV1_convolution import show_image
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.svm import LinearSVC


class KmeanClass:
    # given labels for our images
    names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity',
             'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store',
             'Street', 'Suburb', 'TallBuilding']
    names1 = ['bedroom', 'Coast', 'Forest']

    # dictionaries mapping names to labels and vice versa - used so we can
    # consistently use ndarrays, while preserving the ability to output text labels
    names_dict = {0: 'bedroom', 1: 'Coast', 2: 'Forest', 3: 'Highway', 4: 'industrial',
                  5: 'insidecity', 6: 'kitchen', 7: 'livingroom', 8: 'Mountain',
                  9: 'Office', 10: 'OpenCountry', 11: 'store', 12: 'Street',
                  13: 'Suburb', 14: 'TallBuilding'}

    labels_dict = {'bedroom': 0, 'Coast': 1, 'Forest': 2, 'Highway': 3, 'industrial': 4,
                   'insidecity': 5, 'kitchen': 6, 'livingroom': 7, 'Mountain': 8,
                   'Office': 9, 'OpenCountry': 10, 'store': 11, 'Street': 12,
                   'Suburb': 13, 'TallBuilding': 14}

    NUMBER_OF_CLUSTERS = 500
    NUMBER_OF_PATCHES = 700
    SIZE_OF_PATCH = 8
    MOVE_SIZE = 4
    BATCH_SIZE = 1500

    kmeans = None
    linear_model = None
    row_no = 0
    num_imported = 0
    patches_number = 0

    all_images = []
    patches = np.array([])
    patch_for_image = []
    patch_img = []
    patch_class = []
    x_histograms = []
    y_histograms = []

    def __init__(self):
        print('init')

    def download_images(self):
    # This code will import the image, extract the patches, flatten them and save
    # into an array, along with a label indicating the image type it is applied to.
    # This array needs to be huge - there are 6.5m patches PER CLASS!!!

        for s in self.names1:
            for filename in glob.glob('/home/smart/PycharmProjects/comp_vision_3/training/' + s + '/*.jpg'):
                img = np.array(cv2.imread(filename, 0).astype(float), dtype=np.float)
                img = preprocessing.normalize(img)
                img = preprocessing.scale(img)
                self.all_images.append(img)
                #print('class ', filename, 'image ', s)
                patch_for_image = extract_patches_2d(img, (self.SIZE_OF_PATCH, self.SIZE_OF_PATCH),
                                                     max_patches=self.NUMBER_OF_PATCHES, random_state=37)
                sh = patch_for_image.shape
                patch_for_image = np.reshape(patch_for_image, (sh[0], sh[1]*sh[2]))
                if len(self.patches)>0:
                    self.patches = np.concatenate((self.patches, patch_for_image), axis=0)
                else:
                    self.patches = patch_for_image
                self.patches_number += sh[0]
                #print('patches number ', str(self.patches_number))


    def kmean_clustering(self):
        output_file = open('/home/smart/PycharmProjects/comp_vision_3/data/kmean_big.pkl', 'wb')
        # self.kmeans = KMeans(self.NUMBER_OF_CLUSTERS, max_iter = 100)
        # mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,
        #              n_init=10, max_no_improvement=10, verbose=0)
        self.kmeans = MiniBatchKMeans(init='k-means++', n_clusters=self.NUMBER_OF_CLUSTERS,
                                      batch_size=self.BATCH_SIZE, n_init=10, max_no_improvement=10,
                                      verbose=0)
        self.kmeans = self.kmeans.fit(self.patches)
        pickle.dump(self.kmeans, output_file, -1)
        output_file.close()

    def load_data(self):
        input_file = open('/home/smart/PycharmProjects/comp_vision_3/data/kmean_big.pkl', 'rb')
        self.kmeans = pickle.load(input_file)
        input_file.close()

    def create_histograms(self):
        for s in self.names1:
            for filename in glob.glob('/home/smart/PycharmProjects/comp_vision_3/training/' + s + '/*.jpg'):
                img = np.array(cv2.imread(filename, 0).astype(float), dtype=np.float)
                img = preprocessing.normalize(img)
                img = preprocessing.scale(img)
                self.all_images.append(img)
                print('class ', filename, 'image ', s)
                patch_for_image = extract_patches_2d(img, (self.SIZE_OF_PATCH, self.SIZE_OF_PATCH),
                                                     max_patches=self.NUMBER_OF_PATCHES, random_state=37)
                sh = patch_for_image.shape
                patch_for_image = np.reshape(patch_for_image, (sh[0], sh[1] * sh[2]))
                if len(self.patches) > 0:
                    self.patches = np.concatenate((self.patches, patch_for_image), axis=0)
                else:
                    self.patches = patch_for_image
                labels = []
                for each_patch in patch_for_image:
                    labels.append(kmean_object.kmeans.predict([each_patch]))
                histogram = np.histogram(labels, range(self.NUMBER_OF_CLUSTERS))
                self.x_histograms.append(histogram[0])
                self.y_histograms.append(self.labels_dict[s])

        output_file = open('/home/smart/PycharmProjects/comp_vision_3/data/x_histograms.pkl', 'wb')
        pickle.dump(self.x_histograms, output_file, -1)
        output_file.close()
        output_file = open('/home/smart/PycharmProjects/comp_vision_3/data/y_histograms.pkl', 'wb')
        pickle.dump(self.y_histograms, output_file, -1)
        output_file.close()

    def linear_classifier(self):
        print ('linear')


        # partition the data into training and testing splits, using 75%
        # of the data for training and the remaining 25% for testing
        print("[INFO] constructing training/testing split...")
        (trainData, testData, trainLabels, testLabels) = train_test_split(
            np.array(self.x_histograms), self.y_histograms, test_size=0.2, random_state=58)

        # train the linear regression clasifier
        print("[INFO] training Linear SVM classifier...")
        self.linear_model = LinearSVC()
        self.linear_model.fit(trainData, trainLabels)

        output_file = open('/home/smart/PycharmProjects/comp_vision_3/data/linear_model.pkl', 'wb')
        pickle.dump(self.linear_model, output_file, -1)
        output_file.close()

        # evaluate the classifier
        print("[INFO] evaluating classifier...")
        predictions = self.linear_model.predict(testData)
        print(classification_report(testLabels, predictions))

    def prediction(self):
        s = ''
        num_img = 0
        for filename in glob.glob('/home/smart/PycharmProjects/comp_vision_3/testing/*.jpg'):
            img = np.array(cv2.imread(filename, 0).astype(float), dtype=np.float)
            img = preprocessing.normalize(img)
            img = preprocessing.scale(img)
            histogram = self.histogram_plot(img)
            t = int(kmean_object.linear_model.predict([histogram[0]])[0])
            s += self.names_dict[t] + '\n'
            print (filename, ' num_img = ', str(num_img))
            num_img += 1
        output_file = open('/home/smart/PycharmProjects/comp_vision_3/data/test.txt', 'w')
        output_file.write(s)
        output_file.close()

    def histogram_plot(self, img):
        patch_for_image = extract_patches_2d(img, (self.SIZE_OF_PATCH, self.SIZE_OF_PATCH),
                                             max_patches=self.NUMBER_OF_PATCHES, random_state=37)
        sh = patch_for_image.shape
        patch_for_image = np.reshape(patch_for_image, (sh[0], sh[1] * sh[2]))
        labels = []
        for each_patch in patch_for_image:
            labels.append(kmean_object.kmeans.predict([each_patch]))
        histogram = np.histogram(labels, range(self.NUMBER_OF_CLUSTERS))
        return (histogram)

    def load_histograms(self):
        input_file = open('/home/smart/PycharmProjects/comp_vision_3/data/x_histograms.pkl', 'rb')
        self.x_histograms = pickle.load(input_file)
        input_file.close()
        input_file = open('/home/smart/PycharmProjects/comp_vision_3/data/y_histograms.pkl', 'rb')
        self.y_histograms = pickle.load(input_file)
        input_file.close()

    def load_linear_classifier(self):
        input_file = open('/home/smart/PycharmProjects/comp_vision_3/data/linear_model.pkl', 'rb')
        self.linear_model = pickle.load(input_file)
        input_file.close()

kmean_object = KmeanClass()

#print('download pictures, make patches')
#kmean_object.download_images()
#print('finish downloading')

#print('clusterization')
#k.kmean_clustering()

#print('finish clustering')

# just load data for kmeans classifier

print ('load k-mean data')
kmean_object.load_data()
print ('load complete')

#print('create histograms for images')
#kmean_object.create_histograms()
#print ('histograms are created')

print('load histograms')
kmean_object.load_histograms()
print ('finish loading histograms')

print('linear classifier')
kmean_object.linear_classifier()
print ('linear classifier created')

print('linear classifier load')
kmean_object.load_linear_classifier()
print ('linear classifier loaded')

print ('prediction ')
kmean_object.prediction()
print ('end')

