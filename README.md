# OCR Hiring test for Teko
## A OCR system that be developed to extract the texts from the [image](https://github.com/t3min4l/ocr/blob/master/20000-leagues-006.jpg)
### Installation:
- ```pip install -r requiremens.txt```
### Prepare CRNN model:
- ```chmod 777 prepare.sh```
- ```./prepare.sh```

Using [CRNN model](https://arxiv.org/pdf/1507.05717.pdf) implemented in Pytorch

### Usage:
- Using CRNN to extract text from image: 
	- ```python text_extractor.py```
- Using Tesseract from Google to extract text from image: 
	- ```sudo apt-get install tesseract-ocr```
	- ```python tesseract.py```
