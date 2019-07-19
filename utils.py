import os
import sys
import cv2
import numpy as np

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
    dilation = cv2.dilate(gray,kernel,iterations = 1)
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
		if w >= 5 and h >= 15:
			cv2.rectangle(line_cp1, (x,y), (x+w,y+h), (0,0,0),3)
	res1 = find_bounding_box(line_cp1)
	words = list()
	for (c,_) in res1:
		(x,y,w,h) = cv2.boundingRect(c)
		if w >= 10 and h >= 20:
			words.append(line_cp3[y:y+h, x:x+w])
			cv2.rectangle(line_cp2, (x,y), (x+w, y+h), (0,0,0), 1)

	return words




img = cv2.imread(img_path)
words = segment_words(img)
for word in words:
	display_image(word)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# display_image(img_gray)
# display_image(img_gray)
ret, thresh = cv2.threshold(img_gray,75,240,cv2.THRESH_BINARY)
thresh_cp = thresh.copy()
display_image(thresh)

img1 = thresh.copy()
img2 = thresh.copy()
img3 = thresh.copy()

results = list()
res = find_bounding_box(img1)

for (c, _) in res:
	(x,y,w,h) = cv2.boundingRect(c)
	if w >= 5 and h >= 15:
		cv2.rectangle(img1, (x,y), (x+w, y+h), (0,0,0), 3)
		results.append((x,y,w,h))
display_image(img1)
res1 = find_bounding_box(img1)
result1 = list()
for idx, (c,_) in enumerate(res1):
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 10 and h >= 20:
        # idx += 1
        # if idx == 7:
        #     letter = img3[y:y+h, x:x+w]
        # else:
        #     letter = img3[y-5:y+h, x-5:x+w]
        # filename = os.path.join(folder_path, 'letter_' + str(idx) + '.png')
        # letters.append(filename)
        # cv2.imwrite(filename, letter)
        result1.append((x,y,w,h))
        cv2.imwrite(img_name + str(idx) +'.png', img3[y:y+h,x:x+w])
        cv2.rectangle(img2, (x,y), (x + w, y+h), (0,0,0) ,1)

display_image(img2)
# cv2.imwrite('segments_result.png', img2)
# res2 = find_bounding_box(img2)
# for (c,_) in res2:
#     (x, y, w, h) = cv2.boundingRect(c)
#     if w >= 25 and h >= 30:
#         # idx += 1
#         # if idx == 7:
#         #     letter = img3[y:y+h, x:x+w]
#         # else:
#         #     letter = img3[y-5:y+h, x-5:x+w]
#         # filename = os.path.join(folder_path, 'letter_' + str(idx) + '.png')
#         # letters.append(filename)
#         # cv2.imwrite(filename, letter)
#         result1.append((x,y,w,h))
#         cv2.rectangle(img3, (x,y), (x + w, y+h), (0,0,0) ,1)
# display_image(img3)
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
# im2, contours, hierarchy = cv2.findContours(thresh,1,2)
# print(len(contours))
# for cnt in contours:
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	cv2.rectangle(thresh, (x,y), (x+w,y+h), (0,255,0), 1)
# display_image(thresh)
# im2_2, contours_2, hierarchy_2 = cv2.findContours(thresh,1, 2)
# for cnt in contours_2:
# 	x,y,w,h = cv2.boundingRect(cnt)
# 	cv2.rectangle(thresh_cp, (x,y), (x+w,y+h), (0,255,0), 1)
# display_image(thresh_cp)