import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from utils import *

import torch 
from PIL import Image
from torch.autograd import Variable
from crnn_pytorch import utils
from crnn_pytorch import dataset
from crnn_pytorch import models

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

img_search_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "*.jpg")
img_path = glob.glob(img_search_path)[0]
print(img_path)

black = [10,10,10]
white = [255,255,255]

img = cv2.imread(img_path)

img = img[:,:int(img.shape[1]*0.5),:]

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img1 = img.copy()
	
ret, thresh = cv2.threshold(img, 60, 240, cv2.THRESH_BINARY_INV)
pts = cv2.findNonZero(thresh)
box = cv2.minAreaRect(pts)

(cx,cy), (w,h), angle = box 
if w>h:
	w,h = h,w
	angle += 90
M = cv2.getRotationMatrix2D((cx,cy), angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (img.shape[1], img.shape[0]))

hist = cv2.reduce(rotated,1, cv2.REDUCE_MAX).reshape(-1)
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

lines = list()
print(len(uppers), len(lowers))
for idx,(yupper, ylower) in enumerate(zip(uppers, lowers)):
	cv2.imwrite(os.path.join(os.path.abspath(os.path.dirname(__file__)), str(idx)+".png") ,img1.copy()[yupper-1:ylower+7,:])
	lines.append(cv2.cvtColor(img1.copy()[yupper-1:ylower+7,:], cv2.COLOR_BGR2RGB))

model_path = "./crnn_pytorch/data/crnn.pth"
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model = models.crnn.CRNN(31,1,37,256)
if torch.cuda.is_available():
	model = models.cuda()
print("Loading CRNN...")
model.load_state_dict(torch.load(model_path))
 
converter = utils.strLabelConverter(alphabet)
transformer = dataset.resizeNormalize((100,32))
for idx, line in enumerate(lines):
	words = segment_words(line)
	for word in words:
		word_pil = Image.fromarray(word)
		word = transformer(word_pil)
		# word = Variable(word)
		if torch.cuda.is_available():
			word = word.cuda()
		word = word.view(1, *word.size())
		image = Variable(image)


