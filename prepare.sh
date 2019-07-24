#!/bin/bash
git clone https://github.com/meijieru/crnn.pytorch.git
mv crnn.pytorch crnn_pytorch
ls
wget https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth -P crnn_pytorch/data/
wget wget --no-check-certificate https://docs.google.com/uc\?export\=download\&id\=1hmtbUQC5HuLb1KOMozNwCKFoAPa56rtx -O ./prediction_model.hdf5
