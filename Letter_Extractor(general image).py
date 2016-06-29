import cv2
import numpy as np

def main(num):
	print num
	img = cv2.imread('Sample Images/sample ('+str(num)+').jpg') 

	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)
	imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

	cv2.imshow('img',img)
	cv2.imshow('imgGray',imgGray)
	cv2.imshow('imgBlurred',imgBlurred)
	cv2.imshow('imgThresh',imgThresh)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

for i in range(1,31):
	main(i)