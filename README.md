# HW4_BIU_DEEP

Implementation of [Shortcut-Stacked Sentence Encoders for Multi-Domain Inference](https://arxiv.org/pdf/1708.02312v2.pdf) paper.

1. Download one of the pretrained word vectors in the following link https://nlp.stanford.edu/projects/glove/

2. Install the requirements:
``pip install -r requirements.txt``

3. To train and evaluate our results on the test set you need to run the following command:

  ``python train.py arg1 arg2 arg3 arg4``

  when: 

  arg1: is the pretrained words vectors path of glove

  arg2: is the learning rate

  arg3: is the BiLSTM layer dimension

  arg4: is the dropout probability 

  for examples:
  ``python train.py glove/glove.840B.300d.txt 0.0004 300 0.1``
