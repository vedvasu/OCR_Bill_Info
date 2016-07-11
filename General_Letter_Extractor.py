import cv2
import numpy as np
import math

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

def letterExtraction(img):

    list_letters = []

    imgCopy = img.copy()
    contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)       # finding contours on the image
                                                                                    # no filtering required all done letter extraction
    for c in contours:

        if cv2.contourArea(c) < 35000:
        
            [intX, intY, intWidth, intHeight] = cv2.boundingRect(c)

            crop = imgCopy[intY:intY+intHeight,intX:intX+intWidth]    
            
            cv2.rectangle(imgCopy,(intX, intY),(intX + intWidth, intY + intHeight),127,2)

            ret,crop = cv2.threshold(crop,127,255,cv2.THRESH_BINARY_INV)

            list_letters.append(crop)

            #cv2.imshow('crop',crop)
            #cv2.imshow('img_extract',imgCopy)
            # cv2.waitKey(0)

    return list_letters


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
    
    '''
    * Basic preprocessing for enhancing the edges for further detection and filtering.
    '''
    ret,imgThresh = cv2.threshold(imgEdges,127,255,cv2.THRESH_BINARY_INV)
    
    erosion = cv2.erode(imgThresh,kernel,iterations = 1)
    erosion = cv2.medianBlur(erosion, 3)
    
    #img1 = cv2.cvtColor(erosion, cv2.COLOR_GRAY2RGB)
    # dilation = cv2.dilate(imgThresh, kernel, iterations = 1)

    # opening = cv2.dilate(erosion, kernel, iterations = 1)
    # closing = cv2.erode(dilation, kernel, iterations = 1)


    # cv2.imshow('img',img)
    # cv2.imshow('imgEdges',imgEdges)
    # cv2.imshow('imgThresh',imgThresh)
    
    # cv2.imshow('dilation',dilation)
    # cv2.imshow('opening',opening)
    # cv2.imshow('closing',closing)
    #cv2.imshow('imgLaplacian',imgLaplacian)
    # cv2.imshow('imgSobelx',imgSobelx)
    # cv2.imshow('imgSobely',imgSobely)
    #cv2.imshow('imgGray',imgGray)
    #cv2.imshow('imgBlurred',imgBlurred)
    #cv2.imshow('imgThresh',imgThresh)
    
    return erosion        
    
def operationsStage2_ContourFiltering(erosion):

    erosionCopy = erosion.copy()
    npaContours, npaHierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #print len(npaContours)
    # for c in npaContours:
    #     if cv2.contourArea(c)>10 and cv2.contourArea(c)<8000: 
    #         [intX, intY, intWidth, intHeight] = cv2.boundingRect(c)

    #         crop = erosionCopy[intY:intY+intHeight,intX:intX+intWidth]    
            
    #         cv2.rectangle(img,(intX, intY),(intX + intWidth, intY + intHeight),(127, 255, 0),1)
    #         # cv2.imshow('crop',crop)
    #         # cv2.waitKey(0)

    pixelpoints = 0
    mask = np.zeros(erosionCopy.shape,np.uint8)
    mask_testing = np.zeros(erosionCopy.shape,np.uint8)
    heri_prev1 = npaHierarchy[0][0][2]
    heri_prev2 = npaHierarchy[0][0][3]


####################################################################################################################################################

    for c in range (0,len(npaContours)):
        flag_wrong_contour = 0
        '''
        * The contour area should be reasonable for the text 
        '''
        if cv2.contourArea(npaContours[c])>10 and cv2.contourArea(npaContours[c])<8000:
            heri_next = npaHierarchy[0][c][3]
            #print cv2.contourArea(npaContours[c]),npaHierarchy[0][c]
            '''
            * First condition for including the contours inside contours as black for letters like R, B(hierarchy used)
            * Else contours are included normally as white
            '''
            if (heri_prev1 - heri_next == 1):

                cv2.drawContours(mask,[npaContours[c]],0,0,-1)
                pixelpoints = cv2.findNonZero(mask)
                
                ############# No longer valid as a black rectangle created already (below condition is always flase)
                # '''
                # * This condition is to remove the points unwanted contours by the edges of image
                # '''
                # for i in range(len(pixelpoints)):
                 
                #     if pixelpoints[i][0][0] == 1 or pixelpoints[i][0][0] == 198 or pixelpoints[i][0][1] == 1 or pixelpoints[i][0][1] == 198:

                #         if erosionCopy[pixelpoints[i][0][1],pixelpoints[i][0][0]] == 255:
                #             cv2.circle(mask,(pixelpoints[i][0][0],pixelpoints[i][0][1]),1,128,-1)
                #             flag_wrong_contour = 1               
                
                # if flag_wrong_contour == 0:   
                #     cv2.drawContours(mask_testing,[npaContours[c]],0,0,-1)
                cv2.drawContours(mask_testing,[npaContours[c]],0,0,-1)        #include this line only if above line is commented


            ### Pixel points of a contour
            else:
                if cv2.contourArea(npaContours[c])>=200:
                    
                    mask = np.zeros(erosionCopy.shape,np.uint8)
                    cv2.drawContours(mask,[npaContours[c]],0,255,-1)
                    pixelpoints = cv2.findNonZero(mask)
                    
                    
                    ############# No longer valid as a black rectangle created already (below condition is always flase)
                    # for i in range(len(pixelpoints)):
                        
                    #     if pixelpoints[i][0][0] == 1 or pixelpoints[i][0][0] == 198 or pixelpoints[i][0][1] == 1 or pixelpoints[i][0][1] == 198:
    
                    #         if erosionCopy[pixelpoints[i][0][1],pixelpoints[i][0][0]] == 255:
                    #             cv2.circle(mask,(pixelpoints[i][0][0],pixelpoints[i][0][1]),1,128,-1)
                    #             flag_wrong_contour = 1               
                    # if flag_wrong_contour == 0:     
                    #     cv2.drawContours(mask_testing,[npaContours[c]],0,255,-1)
                    cv2.drawContours(mask_testing,[npaContours[c]],0,255,-1)     #include this line only if above line is commented
                                
            #cv2.imshow('mask',mask_testing)
            #cv2.imshow('erosionCopy',erosionCopy)

            #print pixelpoints
            #cv2.waitKey(0)
        heri_prev1 = npaHierarchy[0][c][2]    
        heri_prev2 = npaHierarchy[0][c][3]
    
    ret,mask_testing = cv2.threshold(mask_testing,127,255,cv2.THRESH_BINARY_INV)

#################################################################################################################################################################


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

    # cv2.imshow('erosion',erosion)
    # cv2.imshow('erosionCopy',erosionCopy)
    # cv2.imshow('mask',mask_testing)

    return mask_testing


def operationsStage3_nontextContoursFiltering(img_pass1):

    ##### OPERATION AFTER FIRST SET COMPLETED
    
    ''' 
    START HERE!!!!!!!!!!
    * Work after pass one is stopped as a big mistake is found (the contours just on the edge may be letters which were just removed)
    * Removed by creating a rectangle on the boundaries of the edge detection image
    '''

    img_pass1Copy = img_pass1.copy()
    npaContours, npaHierarchy = cv2.findContours(img_pass1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    #print len(npaContours)

    # for c in npaContours:
    #     if cv2.contourArea(c)>10 and cv2.contourArea(c)<8000: 
    #         [intX, intY, intWidth, intHeight] = cv2.boundingRect(c)

    #         crop = img_pass1Copy[intY:intY+intHeight,intX:intX+intWidth]    

    #         cv2.rectangle(img,(intX, intY),(intX + intWidth, intY + intHeight),(127, 255, 0),1)
    #         cv2.imshow('crop',crop)
    #         cv2.waitKey(0)

    r = [0,0]
    flag = 0 
    boundary_points = []
    flag_mid = 0
    for c in range(1,len(npaContours)):
        
        mask = np.zeros(img_pass1Copy.shape,np.uint8)
        cv2.drawContours(mask,npaContours,c,127,1)

        for i in range(0,mask.shape[0]):
            for j in range(0,mask.shape[1]):
                if img_pass1Copy[j,i] == 255 and mask[j,i] == 127:
                    boundary_points.append([j,i])
                    print erosionCopy[j,i]
                    if erosionCopy[j,i] == 0:                                
                        ''' WORKING HERE !!!!!!!!!!!!!!!'''
                        print 'yeah'
                    cv2.circle(erosionCopy,(i,j),1,200,-1)
        
        #     if npaContours[c][i][0][0] == r[0] and npaContours[c][i][0][1] == r[1]: 
        #         flag = 1
        #         print 'bhak'
        #         break    
        #     r = npaContours[c][i][0]

        # cv2.drawContours(mask,[npaContours[c]],0,255,-1)
        # pixelpoints = cv2.findNonZero(mask)
        # for i in range(len(pixelpoints)):
        #     if img_pass1Copy[pixelpoints[i][0][1]+5,pixelpoints[i][0][0]+5] == 255 or img_pass1Copy[pixelpoints[i][0][1]-5,pixelpoints[i][0][0]-5] == 255:
        #         cv2.circle(mask,(pixelpoints[i][0][0],pixelpoints[i][0][1]),1,128,0)
        #print boundary_points
        
        # cv2.imshow('mask',mask)
        # cv2.imshow('erosionCopy',erosionCopy)
        # cv2.waitKey(0)


def setup(img):

    erosion = operationsStage1_BasicOperations(img)
    img_pass1 = operationsStage2_ContourFiltering(erosion)
    result = img_pass1.copy()
    #operationsStage3_nontextContoursFiltering(result)
    letters_array = letterExtraction(img_pass1)

    '''
    *For the implementation uptill now result contains the best result 
    ''' 
    return result, letters_array


# img = cv2.imread('samples/fb.jpg') 
# out, list_letters = setup(img)
# cv2.imshow('out',out)
# cv2.waitKey(0)