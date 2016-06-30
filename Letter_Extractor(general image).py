import cv2
import numpy as np
import math

def main(num):
	print num
	img = cv2.imread('Sample Images/sample ('+str(num)+').jpg') 

	imgEdges = cv2.Canny(img,100,200)
	#ret,imgThresh1 = cv2.threshold(imgEdges,150,255,cv2.THRESH_BINARY_INV)

	#imgLaplacian = cv2.Laplacian(img,cv2.CV_64F)
	# imgSobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
	# imgSobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
	
	# for i in range(0,img.shape[1]):
	# 	for j in range(0,img.shape[0]):
			
	# 		#b1,g1,r1 = img[j,i]

	# 		imgTransformed[j,i][0] = math.sqrt((imgSobely[j,i][0]*imgSobely[j,i][0]) + (imgSobelx[j,i][0]*imgSobelx[j,i][0]))
	# 		imgTransformed[j,i][1] = math.sqrt((imgSobely[j,i][1]*imgSobely[j,i][1]) + (imgSobelx[j,i][1]*imgSobelx[j,i][1]))
	# 		imgTransformed[j,i][2] = math.sqrt((imgSobely[j,i][2]*imgSobely[j,i][2]) + (imgSobelx[j,i][2]*imgSobelx[j,i][2]))
	
	# print imgTransformed
	
	# #imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
	imgThresh = cv2.adaptiveThreshold(imgEdges,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
	# cv2.imshow('imgTransformed',imgTransformed)
	
	npaContours, npaHierarchy = cv2.findContours(imgThresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	print len(npaContours)
	for c in npaContours:
		[intX, intY, intWidth, intHeight] = cv2.boundingRect(c)

		crop = imgEdges[intY:intY+intHeight,intX:intX+intWidth]    
        
        cv2.rectangle(img,(intX, intY),(intX + intWidth, intY + intHeight),(0, 255, 0),2)

	cv2.imshow('img',img)
	cv2.imshow('imgEdges',imgEdges)
	cv2.imshow('imgThresh',imgThresh)
	#cv2.imshow('imgLaplacian',imgLaplacian)
	# cv2.imshow('imgSobelx',imgSobelx)
	# cv2.imshow('imgSobely',imgSobely)
	#cv2.imshow('imgGray',imgGray)
	#cv2.imshow('imgBlurred',imgBlurred)
	#cv2.imshow('imgThresh',imgThresh)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

for i in range(1,31):
	main(i)