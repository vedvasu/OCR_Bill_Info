import cv2
import numpy as np
import sys
from Letters_Extractor import extractLetter as ex 

##################################### GLOBAL VARIABLES ########################################
source_path = 'Sample Images/sample (21).jpg'                   # test image path
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

    ascii_code = int(input[0][0])                       # trainer returns the value in the form  eg. [[.97]]
    if ascii_code >= 9:                                 # code <= 9 means it is a digit else alphabet
            sys.stdout.write(chr(ascii_code))
    else:
            sys.stdout.write(str(ascii_code)) 
        


def main():

    img = cv2.imread(source_path)
    
    '''
    * Module Letter_Extractor has been used to extract letters as in fonts
    * But due to randomness of images in the logos (variable colour combinations) this procedure
        can not be used to indentify letters for general case
    * A procedure which can extract letter from any general image is required.......Still be be developed 
    '''
    
    contours = ex(source_path).setup(2)                     # Key is 2 for getting contour set
    imgThresh = ex(source_path).preProcessing()

    for c in contours:
        
        crop = imgThresh[c.intRectY:c.intRectY+c.intRectHeight,c.intRectX:c.intRectX+c.intRectWidth]    
        
        cv2.rectangle(img,(c.intRectX, c.intRectY),(c.intRectX + c.intRectWidth, c.intRectY + c.intRectHeight),(0, 255, 0),2)
        crop = cv2.resize(crop, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))

########################### K-nearest trainer call#######################################        
        # knn = K_nearestMethod()
        # displayTrainerResult(knn.test(crop,3))

################################# SVM trainer call ######################################
        sv = SVM_method()        
        displayTrainerResult(sv.test(crop))


        cv2.imshow('IMG',img)
        cv2.imshow('crop',crop)
        cv2.waitKey(0)


main()