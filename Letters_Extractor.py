import cv2
import numpy as np
import operator
import Save_data as sv

#Global Variables
MIN_CONTOUR_AREA = 100
RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 20

class ContourWithData():

    '''
    * This class defines the characterstics of the contour
    * Also defines the parameters which can be adjusted as per requirement to check the vaidity of coutour
    '''

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


class extractLetter():              # extracting the letter from the image

    def __init__(self,imgPath):

        self.img = cv2.imread(imgPath)
        self.imgCopy = self.img
        self.validContoursWithData = []

    def preProcessing(self):

        '''
        * This function is used for converting the image into binary to form contour around the letter
        '''

        imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
        imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

        return imgThresh

    def validContourDetection(self,img):

        '''
        * This function uses contourWithData class to extract the information of the contour
        * mainly draws and extracts the valid contours from all contours array
        '''

        allContoursWithData = []

        npaContours, npaHierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for npaContour in npaContours:

            contourWithData = ContourWithData()                                             # instantiate a contour with data object
            contourWithData.npaContour = npaContour                                         # assign contour to contour with data
            contourWithData.boundingRect = cv2.boundingRect(contourWithData.npaContour)     # get the bounding rect
            contourWithData.calculateRectTopLeftPointAndWidthAndHeight()                    # get bounding rect info
            contourWithData.fltArea = cv2.contourArea(contourWithData.npaContour)           # calculate the contour area
            allContoursWithData.append(contourWithData)

        for contourWithData in allContoursWithData:
            if contourWithData.checkIfContourIsValid():
                self.validContoursWithData.append(contourWithData)

    def sortingValidContours(self):

        '''
        * We have to extract the contours in a fashion as humans read
        * This functiom sorts the contours wrt y and then wrt x
        '''

        contour_linechange = []
        contours_linechange_sorted=[]

        self.validContoursWithData.sort(key = operator.attrgetter("intRectY"))
        intRectY_previous = 20
        for contourWithData in self.validContoursWithData:

            if ((contourWithData.intRectY+contourWithData.intRectHeight)/2 - intRectY_previous)>25:
                contour_linechange.sort(key = operator.attrgetter("intRectX"))

                for contours in contour_linechange:
                    contours_linechange_sorted.append(contours)

                contour_linechange = []

            contour_linechange.append(contourWithData)
            intRectY_previous = (contourWithData.intRectY+contourWithData.intRectHeight)/2

        self.validContoursWithData = contours_linechange_sorted


    def sortingValidContours_Fonts(self):

        '''
        * An alternate function for sorting only for fonts .ttl to jpg converted images
        * Images having only two lines on text
        '''

        contour_linechange = []
        contours_linechange_sorted=[]
        counter = 0
        self.validContoursWithData.sort(key = operator.attrgetter("intRectY"))

        for contourWithData in self.validContoursWithData:

            if counter == 0 and (contourWithData.intRectY + (contourWithData.intRectHeight/2)) > (self.imgCopy.shape[0]/2):
                contour_linechange.sort(key = operator.attrgetter("intRectX"))

                contours_linechange_sorted = contour_linechange

                contour_linechange = []
                counter+=1

            contour_linechange.append(contourWithData)

        contour_linechange.sort(key = operator.attrgetter("intRectX"))
        for contours in contour_linechange:
            contours_linechange_sorted.append(contours)

        self.validContoursWithData = contours_linechange_sorted

    
    def displayAndCrop(self):

        i=1
        for c in self.validContoursWithData:
            
            crop = self.imgCopy[c.intRectY:c.intRectY+c.intRectHeight,c.intRectX:c.intRectX+c.intRectWidth]
            sv.save_data('dataset',i,crop)
            i+=1
            if i > 62:
                break

            #cv2.rectangle(self.imgCopy,(c.intRectX, c.intRectY),(c.intRectX + c.intRectWidth, c.intRectY + c.intRectHeight),(0, 255, 0),2)
            
            #cv2.imshow('Img',cv2.resize(self.imgCopy,(1000,self.imgCopy.shape[0])))
            #cv2.imshow('Letter',cv2.resize(crop,(24,24)))
            #cv2.waitKey(0)
    
    
    def setup(self):

        self.img = self.preProcessing()
        self.validContourDetection(self.img)
        self.sortingValidContours_Fonts()
        self.displayAndCrop()


# for i in range(1,2):
#     print 'Font',i
#     validletter = extractLetter('fonts/cropped_stage1/font ('+str(i)+').jpg')
#     validletter.setup()
