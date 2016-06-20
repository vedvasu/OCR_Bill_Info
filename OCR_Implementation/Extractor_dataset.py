import cv2
import numpy as np

for font_number in range(1,15):
	
	img = cv2.imread('fonts/converted/font ('+ str(font_number) +').jpg')

	counter1 = 0
	counter2 = 0
	for i in range(400,800):
		
		if counter1 == 0 and img[i,3100][0] < 100:
			counter1+=1
			cy1 = i

		if counter1 > 0 and img[i,3000][0] > 100:
			counter2+=1
		
		if counter2 > 0 and img[i,3000][0] < 100: 
			cy2 = i
			
	#cv2.circle(img,(3000,cy1),10,(0,0,255),-1)
	#cv2.circle(img,(3000,cy2),10,(0,0,255),-1)
	#cv2.imshow('img',cv2.resize(img,(1000,800)))
	
	img_cropped = img[cy1+5:cy2-5,250:3000,:]
	#cv2.imshow('img_cropped',img_cropped)
	cv2.imwrite('fonts/cropped_stage1/font ('+ str(font_number) +').jpg',img_cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()