import cv2
import numpy

img = cv2.imread('sample.jpg')
h,w,t = img.shape
img_roi1 = img[400:740,70:800]
img = cv2.resize(img, (h/4,w/2))
cv2.imwrite('img_roi1.jpg',img_roi1)
cv2.imshow('sample',img_roi1)

cv2.waitKey(0)
cv2.destroyAllWindows()