import cv2
import numpy as np
import sys
from Letters_Extractor import extractLetter as ex 

############################################### GLOBAL VARIABLES ###############################
source_path = 'Sample Images/sample (21).jpg'
classification_path = 'saved_data/flattenedImages.txt' 
flattened_path = 'saved_data/classifications.txt'
RESIZED_IMAGE_WIDTH = 24
RESIZED_IMAGE_HEIGHT = 24
kNearest = cv2.KNearest()
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


def displayTrainerResult(input):

    ascii_code = int(input[0][0])
    if ascii_code >= 9:
            sys.stdout.write(chr(ascii_code))
    else:
            sys.stdout.write(str(ascii_code)) 
        

def main():

    img = cv2.imread(source_path)
    contours = ex(source_path).setup(2)                     # Key is 2 for getting contour set
    imgThresh = ex(source_path).preProcessing()

    for c in contours:
        
        crop = imgThresh[c.intRectY:c.intRectY+c.intRectHeight,c.intRectX:c.intRectX+c.intRectWidth]    
        
        cv2.rectangle(img,(c.intRectX, c.intRectY),(c.intRectX + c.intRectWidth, c.intRectY + c.intRectHeight),(0, 255, 0),2)
        crop = cv2.resize(crop, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        
        knn = K_nearestMethod()
        displayTrainerResult(knn.test(crop,3))

        cv2.imshow('IMG',img)
        cv2.imshow('crop',crop)
        cv2.waitKey(0)

main()