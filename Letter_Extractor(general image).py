import cv2
import numpy as np
import math

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

def filterContour(contours):
    print contours

def main(num):
    print num
    img = cv2.imread('Sample Images/sample ('+str(num)+').jpg') 

    imgEdges = cv2.Canny(img,100,200)
    #ret,imgThresh1 = cv2.threshold(imgEdges,150,255,cv2.THRESH_BINARY_INV)

    #imgLaplacian = cv2.Laplacian(img,cv2.CV_64F)
    # imgSobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    # imgSobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    
    # for i in range(0,img.shape[1]):
    #   for j in range(0,img.shape[0]):
            
    #       #b1,g1,r1 = img[j,i]

    #       imgTransformed[j,i][0] = math.sqrt((imgSobely[j,i][0]*imgSobely[j,i][0]) + (imgSobelx[j,i][0]*imgSobelx[j,i][0]))
    #       imgTransformed[j,i][1] = math.sqrt((imgSobely[j,i][1]*imgSobely[j,i][1]) + (imgSobelx[j,i][1]*imgSobelx[j,i][1]))
    #       imgTransformed[j,i][2] = math.sqrt((imgSobely[j,i][2]*imgSobely[j,i][2]) + (imgSobelx[j,i][2]*imgSobelx[j,i][2]))
    
    # print imgTransformed
    
    # imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
    
    #imgThresh = cv2.adaptiveThreshold(imgEdges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    
    ret,imgThresh = cv2.threshold(imgEdges,127,255,cv2.THRESH_BINARY_INV)
    
    erosion = cv2.erode(imgThresh,kernel,iterations = 1)
    erosionCopy = erosion.copy()
    #img1 = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
    # dilation = cv2.dilate(imgThresh, kernel, iterations = 1)

    # opening = cv2.dilate(erosion, kernel, iterations = 1)
    # closing = cv2.erode(dilation, kernel, iterations = 1)
        
    npaContours, npaHierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #print len(npaContours)
    for c in npaContours:
        if cv2.contourArea(c)>10 and cv2.contourArea(c)<8000: 
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(c)

            crop = erosionCopy[intY:intY+intHeight,intX:intX+intWidth]    
            
            cv2.rectangle(img,(intX, intY),(intX + intWidth, intY + intHeight),(127, 255, 0),1)
            # cv2.imshow('crop',crop)
            # cv2.waitKey(0)

    #filterContour(npaContours)
    mask_testing = np.zeros(imgThresh.shape,np.uint8)
    heri_prev1 = npaHierarchy[0][0][2]
    heri_prev2 = npaHierarchy[0][0][3]
    for c in range (0,len(npaContours)):
        if cv2.contourArea(npaContours[c])>10 and cv2.contourArea(npaContours[c])<8000:
            heri_next = npaHierarchy[0][c][3]
            #print cv2.contourArea(npaContours[c]),npaHierarchy[0][c]

            if (heri_prev1 - heri_next == 1):
                #print 'yoo baby'
                cv2.drawContours(mask,[npaContours[c]],0,0,-1)
                cv2.drawContours(mask_testing,[npaContours[c]],0,0,-1)
            ### Pixel points of a contour
            else:
                if cv2.contourArea(npaContours[c])>200:
                    mask = np.zeros(imgThresh.shape,np.uint8)
                    cv2.drawContours(mask,[npaContours[c]],0,255,-1)
                    cv2.drawContours(mask_testing,[npaContours[c]],0,255,-1)
                    pixelpoints = cv2.findNonZero(mask)
            #cv2.imshow('mask',mask)

            #print pixelpoints
            #cv2.waitKey(0)
        heri_prev1 = npaHierarchy[0][c][2]    
        heri_prev2 = npaHierarchy[0][c][3]
    
    ret,mask_testing = cv2.threshold(mask_testing,127,255,cv2.THRESH_BINARY_INV)




    #     for i in range(200):
    #         for j in range(200):
    #             if cv2.contourArea(npaContours[c])>200 and cv2.contourArea(npaContours[c])<8000:
    #                 dist = cv2.pointPolygonTest(npaContours[c],(j,i),True)
    #                 if dist >= 0:
    #                     erosionCopy[i,j] = 127

    # for i in range(200):
    #     for j in range(200):

    #         if erosionCopy[i,j] < 50:
    #             erosionCopy[i,j] = 255

    # print erosion
    # print erosion.shape
    cv2.imshow('img',img)
    cv2.imshow('mask',mask_testing)
    cv2.imshow('imgEdges',imgEdges)
    
    cv2.imshow('imgThresh',imgThresh)
   
    cv2.imshow('erosion',erosion)
    cv2.imshow('erosionCopy',erosionCopy)

    # cv2.imshow('dilation',dilation)
    # cv2.imshow('opening',opening)
    # cv2.imshow('closing',closing)
    #cv2.imshow('imgLaplacian',imgLaplacian)
    # cv2.imshow('imgSobelx',imgSobelx)
    # cv2.imshow('imgSobely',imgSobely)
    #cv2.imshow('imgGray',imgGray)
    #cv2.imshow('imgBlurred',imgBlurred)
    #cv2.imshow('imgThresh',imgThresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

for i in range(1,30):
    main(i)