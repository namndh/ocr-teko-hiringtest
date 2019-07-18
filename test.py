import os
import sys
import cv2
import numpy as np

img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '10.png')

def display_image(img):
	cv2.imshow('img', img)
	cv2.waitKey()


img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
display_image(img_gray)
ret, thresh = cv2.threshold(img_gray,60,240,cv2.THRESH_BINARY)
thresh_cp = thresh.copy()
# display_image(thresh)

# # cv2.imshow('img', img)
# # cv2.waitKey()
# print(img.shape)
# # noise removal
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# display_image(opening)
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=5)
# # display_image(sure_bg)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # display_image(sure_fg)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
im2, contours, hierarchy = cv2.findContours(thresh,1,2)
print(len(contours))
for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(thresh, (x,y), (x+w,y+h), (0,255,0), 1)
display_image(thresh)
# im2_2, contours_2, hierarchy_2 = cv2.findContours(thresh,1, 2)
# for cnt in contours_2:
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	cv2.rectangle(thresh_cp, (x,y), (x+w,y+h), (0,255,0), 1)
# display_image(thresh_cp)