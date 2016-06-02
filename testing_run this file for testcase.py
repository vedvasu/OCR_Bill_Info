import cv2
import numpy as np
import operator
import os
import sys
######################################## Global Variables ###############################################

MIN_CONTOUR_AREA = 120                                                                     
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 20
SZ=20
bin_n = 16 # Number of bins
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
svm_params = dict( kernel_type = cv2.SVM_LINEAR,svm_type = cv2.SVM_C_SVC,C=2.67, gamma=5.383 )

## Uncomment next two lines to view the output as text file.
#f = open('output.txt','w')
#sys.stdout = f

#########################################################################################################


class ContourWithData():

    npaContour = None           # contour
    boundingRect = None         # bounding rect for contour
    intRectX = 0                # bounding rect top left corner x location
    intRectY = 0                # bounding rect top left corner y location
    intRectWidth = 0            # bounding rect width
    intRectHeight = 0           # bounding rect height
    fltArea = 0.0               # area of contour

    def calculateRectTopLeftPointAndWidthAndHeight(self):               # calculate bounding rect info
        [intX, intY, intWidth, intHeight] = self.boundingRect
        self.intRectX = intX
        self.intRectY = intY
        self.intRectWidth = intWidth
        self.intRectHeight = intHeight

    def checkIfContourIsValid(self):                            # this is oversimplified, for a production grade program
        if self.fltArea < MIN_CONTOUR_AREA: return False        # much better validity checking would be necessary
        return True


def main():
    
    allContoursWithData = []
    validContoursWithData = []
    contour_linechange = []
    contours_linechange_sorted=[]

############################################## Loading Trained Data ##############################################
    
    npaFlattenedImages= np.loadtxt("saved_data/flattened_imagesCombined.txt",np.float32) 
    npaClassifications= np.loadtxt("saved_data/classificationsCombined.txt", np.float32)

############################################# PREPROCESSING AND TRAINING #########################################

# PART A: Kneareat Learning Method to train the trainer with the trained Data
    
    kNearest = cv2.KNearest()
    kNearest.train(npaFlattenedImages, npaClassifications)


# PART B : Loading the Image And Getting the Region of Interest
    
    imgTestingNumbers = cv2.imread('sample.jpg')
    imgTestingNumbers = imgTestingNumbers[400:740,70:800]    
    imgTestingNumbers = cv2.resize(imgTestingNumbers, (1000, 600))

# PART C: Image Preprocessing for forming the image as the dataset samples
    
    imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    imgThreshCopy = imgThresh.copy()


# PART D: Detection of Contours in the image (Image is threshed)
    
    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# PART E: Reading Contour Information and Detecting the Valid Contours in list of contours
    
    for npaContour in npaContours:
        
        contourWithData = ContourWithData()                                             # instantiate a contour with data object
        contourWithData.npaContour = npaContour                                         # assign contour to contour with data
        contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
        contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
        contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
        allContoursWithData.append(contourWithData)

    for contourWithData in allContoursWithData:
        if contourWithData.checkIfContourIsValid():
            validContoursWithData.append(contourWithData)


# PART F: Sorting all the contours based on the top left coordiates of the contour (left to right)
    
    validContoursWithData.sort(key = operator.attrgetter("intRectY"))
    intRectY_previous = 20
    for contourWithData in validContoursWithData:
                                                
        if (contourWithData.intRectY - intRectY_previous)>25:   
            contour_linechange.sort(key = operator.attrgetter("intRectX"))

            for contours in contour_linechange:
                contours_linechange_sorted.append(contours)

            contour_linechange = []

        contour_linechange.append(contourWithData)
        intRectY_previous = contourWithData.intRectY 
    
    validContoursWithData = contours_linechange_sorted  
    
    
########################################### Testing, Detection and Recognition ###############################
    
    i=0
    intRectY_previous = 20
    intRectX_previous = 50
    
    for contourWithData in validContoursWithData:
        i+=1 

# PART A: Contour Preprocessing
        cv2.rectangle(imgTestingNumbers,(contourWithData.intRectX, contourWithData.intRectY),(contourWithData.intRectX + contourWithData.intRectWidth, contourWithData.intRectY + contourWithData.intRectHeight),(0, 255, 0),2)
        
        imgROI = imgThresh[contourWithData.intRectY : contourWithData.intRectY + contourWithData.intRectHeight,contourWithData.intRectX : contourWithData.intRectX + contourWithData.intRectWidth]
        imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
        npaROIResized = np.float32(npaROIResized)

#PART B: Knearest Algorithm testing 
        
        retval, npaResults, neigh_resp, dists = kNearest.find_nearest(npaROIResized, k = 3)
        strCurrentChar = int(npaResults[0][0])

#PART C: Output for the user

        #cv2.namedWindow('Result '+str(i),cv2.WINDOW_NORMAL)
        #cv2.imshow('Result '+str(i),imgROI)
        cv2.imshow("imgTestingNumbers", imgTestingNumbers)
        
        if (contourWithData.intRectX - intRectX_previous) > 10:
        	print " ", 
        if (contourWithData.intRectY - intRectY_previous)>25:   
            print
        if strCurrentChar >= 9:
            sys.stdout.write(chr(strCurrentChar))
        else:
            #sys.stdout.write(chr(strCurrentChar))
        	sys.stdout.write(str(strCurrentChar)) 
        

        intRectY_previous = contourWithData.intRectY
        intRectX_previous = contourWithData.intRectX+contourWithData.intRectWidth 
        
        cv2.waitKey(0)

        cv2.destroyAllWindows()
    
    cv2.imshow("imgTestingNumbers", imgTestingNumbers)
    #cv2.imwrite("D:\SimplyLund/img_roi1_result.jpg",imgTestingNumbers)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

main()