import os
import sys
import cv2
import numpy as np
import glob
img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '18.png')
img_name = '18'
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
    dilation = cv2.dilate(gray,kernel,iterations = 1)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blurred, 30, 150)

    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted([(c, cv2.boundingRect(c)[0]) for c in cnts], key=lambda x: x[1])
    return cnts

def segment_words(line):
	line_rgb = line.copy()
	if len(line.shape) >= 3:
		line_gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
		ret, line = cv2.threshold(line_gray, 75, 240, cv2.THRESH_BINARY)

	line_cp1 = line.copy()
	line_cp2 = line.copy()
	line_cp3 = line.copy()
	res = find_bounding_box(line_cp1)
	for (c, _) in res:
		(x,y,w,h) = cv2.boundingRect(c)
		if w >= 5 and h >= 15:
			cv2.rectangle(line_cp1, (x,y), (x+w,y+h), (0,0,0),3)
	res1 = find_bounding_box(line_cp1)
	words = list()
	for (c,_) in res1:
		(x,y,w,h) = cv2.boundingRect(c)
		if w >= 15 and h >= 20:
			words.append(line_rgb[y:y+h, x:x+w])
			display_image(line_rgb[y:y+h, x:x+w])
			cv2.imwrite('word.png', line[y:y+h, x:x+w])
			cv2.rectangle(line_cp2, (x,y), (x+w, y+h), (0,0,0), 1)
	# display_image(line_cp2)
	# cv2.imwrite('page2.png', line_cp2)
	return words


img_search_path = os.path.join(
	os.path.abspath(os.path.dirname(__file__)), "*.jpg")
img_path = glob.glob(img_search_path)[0]
print(img_path)

black = [10, 10, 10]
white = [255, 255, 255]

img = cv2.imread("20000-leagues-006.jpg")

img = img[:, int(img.shape[1]*0.5):, :]
display_image(img)
lines = segment_words(img)
print(len(lines))
