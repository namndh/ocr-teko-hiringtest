from PIL import Image
import pytesseract
import argparse
import cv2
import os

img_path = "./result.png"

# img = cv2.imread(img_path)
text = pytesseract.image_to_string(Image.open(img_path))
print(text)