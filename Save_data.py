import os
import cv2
import numpy

def save_data(path,folder_number,data):

	'''
	* This function is invloked to save data in the memory
	* It uses the path to reach the directory and folder_number as asked by user
	* It automatically searchs the name with which data is to be saved in serial way.
	'''

	path_list = []

	path = path + '/sample ('+str(folder_number)+')/'
	path_list = os.listdir(path)

	# if path_list == []:
	# 	i=9
	# 	cv2.imwrite(path + '1.jpg',data) 

	# else:
		
	# 	for i in range(0,len(path_list)):
	# 		for j in range(0,len(path_list[i])):

	# 				if path_list[i][j] == '.':
	# 					path_list[i] = int(path_list[i][0:j])
	# 					break
	# 	sorted(path_list)
	# 	cv2.imwrite(path + str(len(path_list)])+'.jpg',data)

	cv2.imwrite(path + str(len(path_list)+1)+'.jpg',data)

#save_data('dataset',1,cv2.imread('Sample Images/sample (1).jpg'))
