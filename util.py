import os
import sys
import cv2
import numpy as np
from PIL import Image


img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '13.png')
img_name = '13'
def display_image(img):
	cv2.imshow('img', img)
	cv2.waitKey()

def find_bounding_box(img):
    # img = image.copy()
    if len(img.shape) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # ret, thresh = cv2.threshold(gray, 0, 255, 0)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(gray,kernel,iterations = 5)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 30, 150)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])
    return cnts

def segment_words(line):
	if len(line.shape) >= 3:
		line_gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
		ret, line = cv2.threshold(line_gray, 75, 240, cv2.THRESH_BINARY)

	line_cp1 = line.copy()
	line_cp2 = line.copy()
	line_cp3 = line.copy()
	res = find_bounding_box(line_cp1)
	for (c, _) in res:
		(x,y,w,h) = cv2.boundingRect(c)
		if w > 5 and h > 15:
			cv2.rectangle(line_cp1, (x,y), (x+w,y+h), (0,0,0),5)
	res1 = find_bounding_box(line_cp1)
	words = list()
	for (c,_) in res1:
		(x,y,w,h) = cv2.boundingRect(c)
		if w > 15 and h > 25:
			words.append(line_cp3[y:y+h, x:x+w])
			cv2.rectangle(line_cp2, (x,y), (x+w, y+h), (0,0,0), 1)
	# display_image(line_cp2)
	return words

def display_img(img):
	cv2.imshow('img', img)
	cv2.waitKey()

def equalize(list1, list2):
	len1 = len(list1)
	len2 = len(list2)
	differ = abs(len1-len2)
	if len1 > len2:
		for i in range(differ):
			del list1[-1]
	if len2 > len1:
		for i in range(differ):
			del list2[-1]
	if len(list1) == len(list2):
		return True
	else:
		return False

def extract_lines(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img1 = img.copy()

	ret, thresh = cv2.threshold(img, 60, 240, cv2.THRESH_BINARY_INV)
	pts = cv2.findNonZero(thresh)
	box = cv2.minAreaRect(pts)

	(cx, cy), (w, h), angle = box
	if w > h:
		w, h = h, w
		angle += 90
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	rotated = cv2.warpAffine(thresh, M, (img.shape[1], img.shape[0]))

	hist = cv2.reduce(rotated, 1, cv2.REDUCE_MAX).reshape(-1)
	th = 10
	H, W = img.shape[:2]
	uppers = [y for y in range(H-1) if hist[y] <= th and hist[y+1] > th]
	lowers = [y for y in range(H-1) if hist[y] > th and hist[y+1] <= th]
	print(len(uppers), len(lowers))
	rotated = cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR)

	for y in uppers:
		cv2.line(rotated, (0, y), (W, y), (255, 0, 0), 1)

	for y in lowers:
		cv2.line(rotated, (0, y), (W, y), (0, 255, 0), 1)

	lines = list()
	print(len(uppers), len(lowers))
	for idx, (yupper, ylower) in enumerate(zip(uppers, lowers)):
		# cv2.imwrite(os.path.join(os.path.abspath(os.path.dirname(__file__)), str(idx)+".png") ,img1.copy()[yupper-1:ylower+7,:])
		lines.append(cv2.cvtColor(
			img1.copy()[yupper-1:ylower+7, :], cv2.COLOR_GRAY2RGB))
	
	return lines