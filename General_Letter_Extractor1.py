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

    print(pytesseract.image_to_string(Image.open(path)))


def operationsStage1_BasicOperations(img):

    imgEdges = cv2.Canny(img,100,200)

    '''
    * As normal threshing ignores the color combination as letters as not be detected generally therefore...
        edge dedtection can detect the edges of the shapes and later text can be seperated
    * Creating a boundary the the boundaries of the image to recognise the half cut contours
    '''
    for i in range(0,200,199):
        for j in range(0,200):
            imgEdges[j,i] = 255

    for i in range(0,200,199):
        for j in range(0,200):
            imgEdges[i,j] = 255

    '''
    * Basic preprocessing for enhancing the edges for further detection and filtering.
    '''
    ret,imgThresh = cv2.threshold(imgEdges,127,255,cv2.THRESH_BINARY_INV)
    
    erosion = cv2.erode(imgThresh,kernel,iterations = 1)
    erosion = cv2.medianBlur(erosion, 3)

    cv2.imshow('img',img)
    cv2.imshow('imgEdges',imgEdges)
    cv2.imshow('imgThresh',imgThresh)
    cv2.imwrite('files/5.jpg',img)
    cv2.imwrite('files/4.jpg',imgEdges)    
    return erosion        
    
def operationsStage2_ContourFiltering(erosion):

    erosionCopy = erosion.copy()
    npaContours, npaHierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    '''
    * This part seperated contour on the basis of heirarchy level
    '''
    
    relatives = []
    home_contours = []
    for c in range (0,len(npaContours)):
        
        if cv2.contourArea(npaContours[c]) > 10 and cv2.contourArea(npaContours[c]) < 8000:
            heri_child = npaHierarchy[0][c][2]
            if heri_child != -1:
                relatives.append(c)
            else:
                relatives.append(c)
                home_contours.append(relatives)
                relatives = []
            
#########################################################################################################################################
    
    pixelpoints = 0
    mask_testing = np.zeros(erosionCopy.shape,np.uint8)


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

####################################################################################################################################################
    
    cv2.imshow('erosionCopy',erosionCopy)
    cv2.imshow('mask',mask_testing) 
    cv2.imwrite('mask_testing.jpg',mask_testing)
    cv2.imwrite('files/3.jpg',erosionCopy)
    cv2.imwrite('files/2.jpg',mask_testing)

    return mask_testing


def operationsStage3_nontextContoursFiltering(img_pass1):

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
            #cv2.waitKey(0)

        cv2.imshow('imgPass1Copy',img_pass1Copy)
        #cv2.waitKey(0)

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

    '''
    *For the implementation uptill now result contains the best result 
    '''
    #result = erosion 
    #return result, letters_array

for i in range(48,56):
    img = cv2.imread('samples/sample (' + str(i) + ').jpg') 
    out = setup(img)
    #cv2.imshow('out',out)
    cv2.waitKey(0)