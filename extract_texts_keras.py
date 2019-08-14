import os
import sys
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

from util import *
import string

from PIL import Image
import keras.backend as K 
from keras.models import model_from_json, load_model
from CRNN_models_keras import CRNN, CRNN_STN
from utils_keras import pad_image, resize_image, create_result_subdir
from STN.spatial_transformer import SpatialTransformer


class Namespace(object):
    def __init__(self, initial_data):
        for key in initial_data:
            setattr(self, key, initial_data[key])

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# parser = argparse.ArgumentParser()
# parser.add_argument('--model_path', type=str, default='')
# parser.add_argument('--data_path', type=str, default='')
# parser.add_argument('--gpus', type=int, nargs='*', default=[0])
# parser.add_argument('--characters', type=str, default='0123456789'+string.ascii_lowercase+'-')
# parser.add_argument('--label_len', type=int, default=16)
# parser.add_argument('--nb_channels', type=int, default=1)
# parser.add_argument('--width', type=int, default=200)
# parser.add_argument('--height', type=int, default=31)
# parser.add_argument('--model', type=str, default='CRNN_STN', choices=['CRNN_STN', 'CRNN'])
# parser.add_argument('--conv_filter_size', type=int, nargs=7, default=[64, 128, 256, 256, 512, 512, 512])
# parser.add_argument('--lstm_nb_units', type=int, nargs=2, default=[128, 128])
# parser.add_argument('--timesteps', type=int, default=50)
# parser.add_argument('--dropout_rate', type=float, default=0.25)
# cfg = parser.parse_args()

arguments = {'model_path': './prediction_model.hdf5', 
             'data_path':'',
             'gpus':[0],
             'characters' : '0123456789'+string.ascii_lowercase+'-',
             'label_len' : 16,
             'width' : 200,
             'height' : 31,
             'model' : 'CRNN_STN',
             'nb_channels': 1,
             'conv_filter_size' : [64, 128, 256, 256, 512, 512, 512],
             'lstm_nb_units' : [128,128],
             'timesteps' : 50,
             'dropout_rate' : 0.25
             }

cfg = Namespace(arguments)
print(cfg.conv_filter_size)

_, model = CRNN_STN(cfg)    
model.load_weights(cfg.model_path)

def img_transform(img):
    if img.shape[1] / img.shape[0] < 6.4:
        img = pad_image(img, (cfg.width, cfg.height), cfg.nb_channels)
    else:
        img = resize_image(img, (cfg.width, cfg.height))
    if cfg.nb_channels == 1:
        img = img.transpose([1, 0])
    else:
        img = img.transpose([1, 0, 2])
    img = np.flip(img, 1)
    img = img / 255.0
    if cfg.nb_channels == 1:
        img = img[:, :, np.newaxis]
    return img


def predict_text(model, img):
    y_pred = model.predict(img[np.newaxis, :, :, :])
    shape = y_pred[:, 2:, :].shape
    ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0])*shape[1])[0][0]
    ctc_out = K.get_value(ctc_decode)[:, :cfg.label_len]
    result_str = ''.join([cfg.characters[c] for c in ctc_out[0]])
    result_str = result_str.replace('-', '')
    return result_str

def word_recognition(words, words_imgs, idx, acc):
    predicted_line = []
    for word in words_imgs:
        word = img_transform(word)
        result_str = predict_text(model,word)
        words.append(result_str+ " ")
        predicted_line.append(result_str)
    if len(predicted_line) > 0:
        acc = evaluate(idx, predicted_line, acc)        
        idx += 1
    return idx, acc

img = cv2.imread("./20000-leagues-006.jpg")

img_p1 = img[:,:int(img.shape[1]*0.5), :]
lines_p1 = extract_lines(img_p1)
words_p1 = list()
img_p2 = img[:, int(img.shape[1]*0.5):, :]
words = segment_words(img_p2)
del words[0]
words_p2 = list()

p1_predicted_true, p2_predicted_true = 0,0
line_idx_p1, line_idx_p2 = 0, 0

for idx, line in enumerate(lines_p1):
	words_imgs = segment_words(line)
	line_idx_p1, p1_predicted_true = word_recognition(words_p1, words_imgs, line_idx_p1, p1_predicted_true)
	words_p1.append('\n')

line_idx_p2, p2_predicted_true = word_recognition(words_p2, words, line_idx_p2, p2_predicted_true)

num_of_words_p1 = 0
for line in labels_p1:
	num_of_words_p1 += len(line)
num_of_words_p2 = 0
for line in labels_p2:
	num_of_words_p2 += len(line)

num_of_words_gt = num_of_words_p1 + num_of_words_p2

print("Page 1:\n")
print(''.join(words_p1)+'\n\n')


print("Page 2:\n")
print(''.join(words_p2) + '\n\n')

print("Model accuracy is {:.2f} %".format((p1_predicted_true+p2_predicted_true)/num_of_words_gt*100))


    