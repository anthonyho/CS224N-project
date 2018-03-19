# CS224N_project
Final project for CS224N: Natural Language Processing with Deep Learning

---------------
GETTING STARTED
---------------

To train model:

1. Download and unzip glove data and place at ./data/glove/ from the following link:
   http://nlp.stanford.edu/data/glove.6B.zip
2. Download train and test data and place at ./data/ from the following link:
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
3. Configure experimental setup at ./experiments/run_{nn,rnn}_model.py
4. Train model and evaluate performance by running
   > cd experiments

   > python run_nn_model.py

   or 

   > python run_rnn_model.py
