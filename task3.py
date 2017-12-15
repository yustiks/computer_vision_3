import cv2
import numpy as np
import glob

from scipy import spatial
from sklearn import preprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import pickle
import os



# import os
# debug: imageviewer from previous work
# from CV1_convolution import show_image
from sklearn.feature_extraction.image import extract_patches_2d



class KmeanClass:
    # given labels for our images
    names = ['bedroom', 'Coast', 'Forest', 'Highway', 'industrial', 'insidecity',
             'kitchen', 'livingroom', 'Mountain', 'Office', 'OpenCountry', 'store',
             'Street', 'Suburb', 'TallBuilding']

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
    NUMBER_OF_PATCHES = 20
    NUMBER_OF_PATCHES_BIG = 3100
    SIZE_OF_PATCH = 8
    MOVE_SIZE = 4
    BATCH_SIZE = 1500
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))+'/'

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
    all_histograms =  np.zeros(NUMBER_OF_CLUSTERS)

    def __init__(self):
        print('init')

    def download_images(self):
    # This code will import the image, extract the patches, flatten them and save
    # into an array, along with a label indicating the image type it is applied to.
    # This array needs to be huge - there are 6.5m patches PER CLASS!!!

        for s in self.names:
            for filename in glob.glob(self.FILE_DIRECTORY+'training1/' + s + '/*.jpg'):
                print('error ' ,s)
                img = np.array(cv2.imread(filename, 0).astype(float), dtype=np.float)
#                img = preprocessing.normalize(img)
                #img = preprocessing.scale(img)
                self.all_images.append(img)
                #print('class ', filename, 'image ', s)
                patch_for_image = extract_patches_2d(img, (self.SIZE_OF_PATCH, self.SIZE_OF_PATCH),
                                                     max_patches=self.NUMBER_OF_PATCHES, random_state=37)
                sh = patch_for_image.shape
                patch_for_image = np.reshape(patch_for_image, (sh[0], sh[1]*sh[2]))
                patch_for_image = preprocessing.normalize(patch_for_image)
                patch_for_image = preprocessing.scale(patch_for_image)
                if len(self.patches)>0:
                    self.patches = np.concatenate((self.patches, patch_for_image), axis=0)
                else:
                    self.patches = patch_for_image
                self.patches_number += sh[0]
                #print('patches number ', str(self.patches_number))

    def kmean_clustering(self):
#        output_file = open(self.FILE_DIRECTORY+'data/kmean_big.pkl', 'wb+')
        self.kmeans = KMeans(self.NUMBER_OF_CLUSTERS, max_iter = 100)
        # miniBatch - when there are millions of pictures
        #self.kmeans = MiniBatchKMeans(init='k-means++', n_clusters=self.NUMBER_OF_CLUSTERS,
        #                              batch_size=self.BATCH_SIZE, n_init=10, max_no_improvement=10,
        #                              verbose=0)
        self.kmeans = self.kmeans.fit(self.patches)
 #       pickle.dump(self.kmeans, output_file, -1)
 #       output_file.close()

    def load_data(self):
        input_file = open(self.FILE_DIRECTORY+'data/kmean_big.pkl', 'rb+')
        self.kmeans = pickle.load(input_file)
        input_file.close()


# create the array with all the patches for 1 class
    def create_histograms(self):
        for s in self.names:
            for filename in glob.glob(self.FILE_DIRECTORY+'training1/' + s + '/*.jpg'):
                img = np.array(cv2.imread(filename, 0).astype(float), dtype=np.float)
#                img = preprocessing.normalize(img)
                #img = preprocessing.scale(img)
                self.all_images.append(img)
#                print('class ', filename, 'image ', s)
                patch_for_image = extract_patches_2d(img, (self.SIZE_OF_PATCH, self.SIZE_OF_PATCH),
                                                     max_patches=self.NUMBER_OF_PATCHES_BIG, random_state=37)
                sh = patch_for_image.shape
                patch_for_image = np.reshape(patch_for_image, (sh[0], sh[1] * sh[2]))
                patch_for_image = preprocessing.normalize(patch_for_image)
                patch_for_image = preprocessing.scale(patch_for_image)
                if len(self.patches) > 0:
                    self.patches = np.concatenate((self.patches, patch_for_image), axis=0)
                else:
                    self.patches = patch_for_image

                for each_patch in patch_for_image:
                    num_cl = int(self.labels_dict[s])
                    predict = self.prediction([each_patch])
                    self.histograms[num_cl][predict] += 1


#        output_file = open(self.FILE_DIRECTORY+'data/x_histograms.pkl', 'wb+')
#        pickle.dump(self.x_histograms, output_file, -1)
#        output_file.close()
#        output_file = open(self.FILE_DIRECTORY+'data/y_histograms.pkl', 'wb+')
#        pickle.dump(self.y_histograms, output_file, -1)
#        output_file.close()


    def prediction(self):
        s = ''
        num_img = 0
        for filename in sorted(glob.glob(self.FILE_DIRECTORY+'testing/*.jpg')):
            st_num = filename.index('testing\\')
            st = filename[st_num+8:]
            img = np.array(cv2.imread(filename, 0).astype(float), dtype=np.float)
            # img = preprocessing.normalize(img)
            # img = preprocessing.scale(img)
            histogram = self.histogram_plot(img)
            t = int(kmean_object.cosine_similarity([histogram[0]],self.all_histograms))
            s += st + ' ' + self.names_dict[t] + '\n'
            print (st + ' ' + self.names_dict[t])
            num_img += 1
        output_file = open(self.FILE_DIRECTORY+'run2.txt', 'w+')
        output_file.write(s)
        output_file.close()

    def histogram_plot(self, img):
        patch_for_image = extract_patches_2d(img, (self.SIZE_OF_PATCH, self.SIZE_OF_PATCH),
                                             max_patches=self.NUMBER_OF_PATCHES_BIG, random_state=37)
        sh = patch_for_image.shape
        patch_for_image = np.reshape(patch_for_image, (sh[0], sh[1] * sh[2]))
        patch_for_image = preprocessing.normalize(patch_for_image)
        patch_for_image = preprocessing.scale(patch_for_image)
        labels = np.zeros(15)
        for each_patch in patch_for_image:
            labels[kmean_object.kmeans.predict([each_patch])] += 1
        return (labels)

    def cosine_similarity(self, histogram1, histogram2):
        t = 0
        result = 0
        for i in range(15):
            cosine_sim = 1 - spatial.distance.cosine(histogram1, histogram2)
            if result < cosine_sim:
                result = cosine_sim
                t = i
        return(i)


# this function used when test
    def load_histograms(self):
        input_file = open(self.FILE_DIRECTORY+'data/all_histograms.pkl', 'rb+')
        self.all_histograms = pickle.load(input_file)
        input_file.close()
        input_file = open(self.FILE_DIRECTORY+'data/y_histograms.pkl', 'rb+')
        self.y_histograms = pickle.load(input_file)
        input_file.close()

kmean_object = KmeanClass()

print('download pictures, make patches 1')
kmean_object.download_images()
print('finish downloading')

print('clusterization')
kmean_object.kmean_clustering()
print('finish clustering')

# just load data for kmeans classifier

#print ('load k-mean data')
#kmean_object.load_data()
#print ('load complete')

print('create histograms for images')
kmean_object.create_histograms()
print ('histograms are created')

#print('load histograms')
#kmean_object.load_histograms()
#print ('finish loading histograms')



print ('prediction ')
kmean_object.prediction()
print ('end')