import cv2
import numpy as np
import sys
import General_Letter_Extractor as ex 

##################################### GLOBAL VARIABLES ########################################
source_path = 'samples/fb.jpg'                   # test image path
classification_path = 'saved_data/flattenedImages.txt'          # training set path
flattened_path = 'saved_data/classifications.txt'
RESIZED_IMAGE_WIDTH = 24
RESIZED_IMAGE_HEIGHT = 24
kNearest = cv2.KNearest()                   # For K-nearrest ML method

# For SVM Mavhine Learning Method
SZ=20
bin_n = 16
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
svm_params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C=2.67, gamma=5.383 )
svm = cv2.SVM()
################################################################################################


class K_nearestMethod():

    def __init__(self):
        
        npaFlattenedImages= np.loadtxt("saved_data/flattenedImages.txt",np.float32) 
        npaClassifications= np.loadtxt("saved_data/classifications.txt", np.float32)
        kNearest.train(npaFlattenedImages, npaClassifications)

    def test(self,img,k):

        imgReshaped = img.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        imgReshaped = np.float32(imgReshaped)
        retval, npaResults, neigh_resp, dists = kNearest.find_nearest(imgReshaped, k)

        return npaResults


class SVM_method():

    def __init__(self):
        
        npaFlattenedImages= np.loadtxt("saved_data/flattenedImages.txt",np.float32) 
        npaClassifications= np.loadtxt("saved_data/classifications.txt", np.float32)
        svm.train(npaFlattenedImages, npaClassifications,params = svm_params)

    def test(self,img):

        imgReshaped = img.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

        imgReshaped = np.float32(imgReshaped)
        npaResults = svm.predict_all(imgReshaped)

        return npaResults


def displayTrainerResult(input):

    result = []
    ascii_code = int(input[0][0])                       # trainer returns the value in the form  eg. [[.97]]
    if ascii_code >= 9:                                 # code <= 9 means it is a digit else alphabet
            #sys.stdout.write(chr(ascii_code))
            return(chr(ascii_code))
    else:
            #sys.stdout.write(str(ascii_code)) 
            return(str(ascii_code))


def setup(source_path):

    img = cv2.imread(source_path)
    
    '''
    * Module Letter_Extractor (general) has been used to extract letters as in fonts
                - This module is supposed to give a image having every letter with black on white background
                - Also every letter should be seperate to form seperate contours.

    * But due to randomness of images in the logos (variable colour combinations) this procedure
        can not be used to indentify letters for general case
    * A procedure which can extract letter from any general image is required.......Still be be developed 
    '''

    letters_recognised = []
    imgLetters, listLetters = ex.setup(img)     # image description in first point of above comments
    
    for crop in listLetters:

        crop = cv2.resize(crop, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

########################### K-nearest trainer call#######################################        
        knn = K_nearestMethod()
        letters_recognised.append(displayTrainerResult(knn.test(crop,3)))                # recognising with 3 nearest neighbour

################################# SVM trainer call ######################################
        # sv = SVM_method()        
        # displayTrainerResult(sv.test(crop)

        #cv2.imshow('img',img)
        #cv2.imshow('crop',crop)
        #cv2.waitKey(0)

    return letters_recognised

#setup(source_path)