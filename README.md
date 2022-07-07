# HW4_BIU_DEEP

First you need to download one of the pretrained word vectors in the following link https://nlp.stanford.edu/projects/glove/

And need to install the requirements:
``pip install -r requirements.txt``

To train and evaluate our results on the test set you need to run the following command:

``python train.py arg1 arg2 arg3 arg4``

when: 

arg1: is the pretrained words vectors path of glove

arg2: is the learning rate

arg3: is the BiLSTM layer dimension

arg4: is the dropout probability 

for examples:
``python train.py glove/glove.840B.300d.txt 0.0004 300 0.1``
