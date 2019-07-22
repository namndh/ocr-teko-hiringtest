from PIL import Image
import pytesseract
import argparse
import cv2
import os

img_path = "./20000-leagues-006.jpg"

def pre_processing(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# img1 = img.copy()

	ret, thresh = cv2.threshold(img, 60, 240, cv2.THRESH_BINARY_INV)
	pts = cv2.findNonZero(thresh)
	box = cv2.minAreaRect(pts)

	(cx, cy), (w, h), angle = box
	if w > h:
		w, h = h, w
		angle += 90
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	rotated = cv2.warpAffine(thresh, M, (img.shape[1], img.shape[0]))
	return rotated

img = cv2.imread(img_path)
img_p1 = img[:,:int(img.shape[1]*0.5),:]
img_p2 = img[:,int(img.shape[1]*0.5):,:]

rotated_p1 = pre_processing(img_p1)

rotated_p2 = pre_processing(img_p2)

# img = cv2.imread(img_path)
print("Page 1:\n\n")
text_p1 = pytesseract.image_to_string(Image.fromarray(rotated_p1))
print(text_p1)
text_p2 = pytesseract.image_to_string(Image.fromarray(rotated_p2))
print("Page 2:\n\n")
print(text_p2)