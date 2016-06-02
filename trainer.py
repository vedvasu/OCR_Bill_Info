import cv2
import numpy as np
import sys
import os

class preProcessing():
    def __init__(self,image_file):
        self.npaFlattenedImages=np.empty((0,400))
        self.image_file=cv2.imread(image_file)
        #self.preProcessImage()

    def preProcessImage(self):

        MIN_CONTOUR_AREA = 0
        RESIZED_IMAGE_WIDTH = 20
        RESIZED_IMAGE_HEIGHT = 20

        cv2.imshow('img',self.image_file)
        self.imgGray = cv2.cvtColor(self.image_file, cv2.COLOR_BGR2GRAY)
        self.imgBlurred = cv2.GaussianBlur(self.imgGray, (5,5), 0)
        self.imgThresh = cv2.adaptiveThreshold(self.imgBlurred,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

        cv2.imshow('imgGray',self.imgGray)
        cv2.imshow('imgBlurred',self.imgBlurred)
        cv2.imshow('imgThresh',self.imgThresh)

        self.imgThreshCopy = self.imgThresh.copy() 

        self.npaContours, self.npaHierarchy = cv2.findContours(self.imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(self.imgThreshCopy,self.npaContours,-1,(255,0,0),1)
        

        check=1
        for self.npaContour in self.npaContours:
            #print cv2.contourArea(self.npaContour)
            if cv2.contourArea(self.npaContour) > MIN_CONTOUR_AREA:
                check=check+1
                [intX, intY, intW, intH] = cv2.boundingRect(self.npaContour)

                cv2.rectangle(self.imgThreshCopy, (intX, intY),(intX+intW,intY+intH),(255, 0, 255),1)
                self.imgROI = self.imgThresh[intY:intY+intH, intX:intX+intW]
                #print self.imgROI.shape
                self.imgROIResized = cv2.resize(self.imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                
                cv2.imshow("ROI",self.imgROI)
                cv2.imshow("ROIResized",self.imgROIResized)              
                npaFlattenedImage = self.imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                self.npaFlattenedImages = np.append(self.npaFlattenedImages, npaFlattenedImage, 0)
                
                if check>2:
                    #print check
                    sys.exit(0)
        #cv2.imshow('imgThreshCopy',self.imgThreshCopy)
        #print self.npaFlattenedImages

        #print "training complete"
        return self.npaFlattenedImages
for i in range(1,3):
    #if i==500:
    #print "training complete"
    #text =obj=preProcessing('samples/sample(0)/node'+str(i)+'.jpg')
    #print i
    obj=preProcessing('samples/sample(1)/node'+str(i)+'.jpg')
    text = obj.preProcessImage()
    #print text
    np.savetxt('flat_text/flattened_images'+str(i)+'.txt', text)
cv2.waitKey(0)
cv2.destroyAllWindows()
        
