import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from util import *

import torch 
from PIL import Image
from torch.autograd import Variable
from crnn_pytorch import utils
import crnn_pytorch.models.crnn as crnn
from crnn_pytorch import dataset

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

def load_crnn():
	model_path = "./crnn_pytorch/data/crnn.pth"
	alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
	model = crnn.CRNN(32, 1, 37, 256)

	if torch.cuda.is_available():
		model = model.cuda()
	print("Loading CRNN...")
	model.load_state_dict(torch.load(model_path))
	converter = utils.strLabelConverter(alphabet)
	transformer = dataset.resizeNormalize((100, 32))
	model.eval()
	return model, converter, transformer

def word_recognition(words, words_imgs):
	for word in words_imgs:
		word_pil = Image.fromarray(word).convert('L')
		word = transformer(word_pil)
		# word = Variable(word)
		if torch.cuda.is_available():
			word = word.cuda()
		word = word.view(1, *word.size())
		word = Variable(word)
		
		preds = model(word)
		_, preds = preds.max(2)
		preds = preds.transpose(1, 0).contiguous().view(-1)
		preds_size = Variable(torch.IntTensor([preds.size(0)]))
		sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
		words.append(sim_pred + " ")

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

black = [10,10,10]
white = [255,255,255]

img = cv2.imread("./20000-leagues-006.jpg")

img = img[:,:int(img.shape[1]*0.5),:]



lines = extract_lines(img)
print(len(lines))



words_p1 = list()
model, converter, transformer = load_crnn()

for idx, line in enumerate(lines):
	words_imgs = segment_words(line)
	word_recognition(words_p1, words_imgs)
	words_p1.append('\n')



print("Page 1:\n\n")
print(''.join(words_p1)+'\n\n')
print("Page 2:\n\n")

img2 = cv2.imread("./20000-leagues-006.jpg")
img2 = img2[:, int(img2.shape[1]*0.5):, :]
# display_image(img2)
words = segment_words(img2)
del words[0]
words_p2 = list()

word_recognition(words_p2, words)
print(''.join(words_p2))


