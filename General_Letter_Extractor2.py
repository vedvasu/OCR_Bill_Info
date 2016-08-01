import cv2
import numpy as np
import math
try:
    import Image
except ImportError:
    from PIL import Image
import cv2
import pytesseract

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

def tesseract_ocr(path):
    
    '''
    * TESSERACT-ORC based detection of letters
    '''
    
    detection = pytesseract.image_to_string(Image.open(path))
    print detection if len(detection) > 0 else "No letters detected"


def operationsStage1_BasicOperations(img):

    '''
    * PART A: Increasing extra 5 pixel in the shape of image for removing unwanted contours by the edges
    * As normal threshing ignores the color combination as letters as not be detected generally therefore...
        edge dedtection can detect the edges of the shapes and later text can be seperated
    * Creating a boundary the the boundaries of the image to recognise the half cut contours
    
    '''
    img_scaledup = np.zeros((img.shape[0]+5,img.shape[1]+5,3),np.uint8)
    
    for i in range(1,img_scaledup.shape[1]-1):
        for j in range(1,img_scaledup.shape[0]-1):

            img_scaledup[j,i] = 255 if i < 4 or j < 4 or i > img.shape[1]-1 or j > img.shape[0]-1 else img[j,i]

    imgEdges = cv2.Canny(img_scaledup,100,200)
    

    '''
    * Part B: Basic preprocessing for enhancing the edges for further detection and filtering.
    '''
    ret,imgThresh = cv2.threshold(imgEdges,127,255,cv2.THRESH_BINARY_INV)
    erosion = cv2.erode(imgThresh,kernel,iterations = 1)
    erosion = cv2.medianBlur(erosion, 3)

    cv2.imshow('img',img)
    cv2.imshow('new_Scaling',img_scaledup)
    cv2.imshow('imgEdges',imgEdges)
    cv2.imshow('imgThresh',imgThresh)
    
    return erosion        
    
def operationsStage2_ContourFiltering(erosion):

    erosionCopy = erosion.copy()
    npaContours, npaHierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    '''
    * Part A: This part seperated contour on the basis of heirarchy level
    * Counters connected by some relation are stored in the list relatives
    * all relatives arrays are stored in home_contours list
    '''
    relatives = []
    home_contours = []
    for c in range (0,len(npaContours)): 

        if cv2.contourArea(npaContours[c]) > 30 and cv2.contourArea(npaContours[c]) < 8000:

            heri_child = npaHierarchy[0][c][2]
            heri_parent = npaHierarchy[0][c][3]

            maxi = npaContours[c][0][0][0]
            mini = npaContours[c][0][0][0]

            for k in range(len(npaContours[c])):
                if npaContours[c][k][0][0] > maxi:
                    maxi = npaContours[c][k][0][0]
                if npaContours[c][k][0][1] > maxi:
                    maxi = npaContours[c][k][0][1]
                if npaContours[c][k][0][0] < mini:
                    mini = npaContours[c][k][0][0]
                if npaContours[c][k][0][1] < mini:
                    mini = npaContours[c][k][0][1]

            if maxi != 197 and mini != 2:
                if heri_child != -1:
                    relatives.append(c)
                else:
                    relatives.append(c)
                    home_contours.append(relatives)
                    relatives = []
           
    pixelpoints = 0
    mask_testing = np.zeros(erosionCopy.shape,np.uint8)

    '''
    * Part B: Seperation on the basis og above heirarchy level
    '''

    for i in range(len(home_contours)):

        if len(home_contours[i]) == 1:
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][0]]],0,255,-1)

        
        elif len(home_contours[i]) == 2:

            if npaHierarchy[0][home_contours[i][0]][3] == -1:
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][0]]],0,255,-1)
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][0]]],0,0,-1)    
            
            else:
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][1]]],0,255,-1)

        else:

            if npaHierarchy[0][home_contours[i][0]][3] == -1:
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][0]]],0,255,-1)
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][len(home_contours[i])-1]]],0,0,-1)    
            
            else:
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][1]]],0,255,-1)
                cv2.drawContours(mask_testing,[npaContours[home_contours[i][len(home_contours[i])-1]]],0,0,-1)

    ret,mask_testing = cv2.threshold(mask_testing,127,255,cv2.THRESH_BINARY_INV)

    cv2.imshow('erosionCopy',erosionCopy)
    cv2.imshow('mask',mask_testing) 

    return mask_testing


def operationsStage3_nontextContoursFiltering(img_pass1):

    '''
    * Part A: Removes unwanted contours
    * Still in progress for better detection and more better categorisation of contours
    * Methods based on area, aspect ratio and contour location are applied
    '''

    img_pass1Copy = img_pass1.copy()
    npaContours, npaHierarchy = cv2.findContours(img_pass1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for i in range(1,len(npaContours)):
       
        [intX, intY, intWidth, intHeight] = cv2.boundingRect(npaContours[i])
        area1 = intHeight * intWidth

        for j in range(1,len(npaContours)):

            [X, Y, Width, Height] = cv2.boundingRect(npaContours[j])
            cx = X + (Width/2)
            cy = Y + (Height/2)
            area2 = Width * Height
            
            if intX < cx < intX + intWidth and intY < cy < intY + intHeight and i != j and area1 > area2: 
                cv2.drawContours(img_pass1Copy,npaContours,j,255,-1)

    cv2.imshow('imgPass1Copy',img_pass1Copy)
    cv2.imwrite('files/1.jpg',img_pass1Copy)
    
    return img_pass1Copy

def setup(img):

    erosion = operationsStage1_BasicOperations(img)
    img_pass1 = operationsStage2_ContourFiltering(erosion)
    img_pass2 = operationsStage3_nontextContoursFiltering(img_pass1)
    cv2.imwrite('temp.jpg',img_pass2)
    tesseract_ocr('temp.jpg')
    result = img_pass2.copy()

    return result

img = cv2.imread('samples/sample (55).jpg') 
out = setup(img)
cv2.waitKey(0)