# OCR Hiring test for Teko
## A OCR system that be developed to extract the texts from the [image](https://github.com/t3min4l/ocr/blob/master/20000-leagues-006.jpg)
### Installation:
- ```pip install -r requiremens.txt```
### Prepare CRNN model:
- ```chmod 777 prepare.sh```
- ```./prepare.sh```

Using [CRNN model](https://arxiv.org/pdf/1507.05717.pdf) implemented in:
- Pytorch by [/meijieru](https://github.com/meijieru/crnn.pytorch)
- Keras by [/kurapan](https://github.com/kurapan/CRNN)

### Usage:
- Active environment
- Using CRNN in Pytorch to extract text from image: 
	- ```python extract_texts_pytorch.py ```
- Using CRNN in Keras to extract text from given image:
	- ```python extract_texts_keras.py```
