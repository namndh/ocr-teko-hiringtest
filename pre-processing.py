import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):
	cv2.imshow('img', img)
	cv2.waitKey()

img_search_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "*.jpg")
img_path = glob.glob(img_search_path)[0]
print(img_path)

black = [10,10,10]
white = [255,255,255]

img = cv2.imread(img_path)

img = img[:,:int(img.shape[1]*0.5),:]

# for x in range(0, img.shape[1]):
# 	for y in range(0, img.shape[0]):
# 		channel_xy = img[y,x]
# 		if all(channel_xy == black):
# 			img[y,x] = white

# display_img(img)

# cv2.imshow('img',img)
# cv2.waitKey()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = img < 200

	
ret, thresh = cv2.threshold(img, 60, 240, cv2.THRESH_BINARY_INV)
pts = cv2.findNonZero(thresh)
box = cv2.minAreaRect(pts)
# box = cv2.boxPoints(box)
# box = np.int0(box)
# cv2.drawContours(thresh, [box], 0, (255,255,255), 1)
# print(box)
(cx,cy), (w,h), angle = box
if w>h:
	w,h = h,w
	angle += 90
M = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
# display_img(thresh)
rotated = cv2.warpAffine(thresh, M, (img.shape[1], img.shape[0]))
# display_img(rotated)
# hist = cv2.reduce(rotated)
# print(pts)
# display_img(thresh)

hist = cv2.reduce(rotated,1, cv2.REDUCE_MAX).reshape(-1)
# print(hist)
th = 10
H,W = img.shape[:2]
uppers = [y for y in range(H-1) if hist[y]<=th and hist[y+1]>th]
lowers = [y for y in range(H-1) if hist[y]>th and hist[y+1]<=th]
print(len(uppers), len(lowers))
rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)
for y in uppers:
    cv2.line(rotated, (0,y), (W, y), (255,0,0), 1)

for y in lowers:
    cv2.line(rotated, (0,y), (W, y), (0,255,0), 1)

display_img(rotated)

# cv2.imwrite("result.png", rotated)