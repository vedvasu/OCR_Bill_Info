import cv2
import numpy as np
import ImageFont
import Image
import ImageDraw

#img = cv2.imread('fonts/AA.TTF')
#cv2.imshow('img',img)
font = ImageFont.truetype('fonts/AA.TTF')
print font

#img.save("image.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()