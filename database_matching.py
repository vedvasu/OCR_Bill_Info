import cv2
import numpy
import ML_Testing as ml
import sys

# f = open('results/results 50 samples.txt','w')
# sys.stdout = f

################## Global Variable #############################
source_path = 'samples/fb.jpg'
################################################################
input_querry = cv2.imread(source_path)
letters_source = ml.setup(source_path)

print 'Letters recognised from source image are: ', letters_source

#################### Testing the input query with last 50 images in the database ######################
print 
print 'Checking input querry in dataset data set...'

for i in range(1,51):

	good = 0
	letters_data = ml.setup('test_cases/data (' + str(i) + ').jpg')
	print 'file name : data (' + str(i) + ').jpg'
	print 'Letters recognised: ',letters_data

	for l1 in letters_data:
		for l2 in letters_source:
			if l1 == l2:
				good+=1

	if good > 0:
		print "Your company named ",
		for l in letters_source:
			print letters_source,
		print "may resemble this logo"


		cv2.imwrite('results/data (' + str(i) + ').jpg',cv2.imread('test_cases/data (' + str(i) + ').jpg'))

	print 