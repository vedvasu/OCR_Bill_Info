import cv2
import numpy as np

def regionOI_extractor(numberOfFonts):
	
	'''
	* This Function gets the region of interest from the already converted .ttf file to .jpg file(A4 landscape)
	* The region of interest contains all the alphabets (lower and uppercase), digits and special charactors.
	* The ROI is written in the memory and furthur used for letters extraction. 
	'''

	for font_number in range(1,numberOfFonts+1):			# loop runs for all the font image file

		img = cv2.imread('fonts/converted/font ('+ str(font_number) +').jpg')

		# The font image has the region between the two lines 1 pixel of width each
		# The ROI is extracted in between these lines with following algorithm

		counter1 = 0
		counter2 = 0
		for i in range(400,800):
			
			if counter1 == 0 and img[i,3100][0] < 100:
				counter1+=1
				cy1 = i

			if counter1 > 0 and img[i,3100][0] > 100:
				counter2+=1
			
			if counter2 > 0 and img[i,3100][0] < 100: 
				cy2 = i
				
		#cv2.circle(img,(3000,cy1),10,(0,0,255),-1)
		#cv2.circle(img,(3000,cy2),10,(0,0,255),-1)
		#cv2.imshow('img',cv2.resize(img,(1000,800)))
		
		img_cropped = img[cy1+5:cy2-5,250:3000,:]
		#cv2.imshow('img_cropped',img_cropped)
		
		print 'saving font '+str(font_number)
		cv2.imwrite('fonts/cropped_stage1/font ('+ str(font_number) +').jpg',img_cropped)

def letterExtractor():  

	'''
	* This function extracts each letter from the ROI obtained from the regionOI_extractor()
	* The letter are stored in the memory in the from of proper dataset
	* This dataset will be used for training the machine learining algorithm
	'''

	for font_number in range(1,numberOfFonts+1):			# loop runs for all the ROI image file

		img = cv2.imread('fonts/cropped_stage1/font ('+ str(font_number) +').jpg')

		


regionOI_extractor(34)
letterExtractor(34)


cv2.waitKey(0)
cv2.destroyAllWindows()