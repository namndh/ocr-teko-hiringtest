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



img = cv2.imread("./20000-leagues-006.jpg")

img_p1 = img[:,:int(img.shape[1]*0.5),:]
lines_p1 = extract_lines(img_p1)

img_p2 = img[:, int(img.shape[1]*0.5):, :]
words = segment_words(img_p2)
del words[0]
words_p2 = list()



words_p1 = list()
model, converter, transformer = load_crnn()

for idx, line in enumerate(lines_p1):
	words_imgs = segment_words(line)
	# for word_img in words_imgs:
	# 	display_image(word_img)
	word_recognition(words_p1, words_imgs)
	words_p1.append('\n')

word_recognition(words_p2, words)


print("Page 1:\n\n")
print(''.join(words_p1)+'\n\n')
print("Page 2:\n\n")
print(''.join(words_p2))


