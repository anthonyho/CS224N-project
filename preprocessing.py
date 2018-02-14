#!/usr/bin/env python
import pandas as pd
import nltk
import numpy as np

def load_data(file_name):
    return pd.read_csv(file_name, encoding='utf-8')

def tokenize_single(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens

def tokenize_df(df, column='comment_text'):
    list_list_tokens = []
    for index, item in df[column].iteritems():
        list_list_tokens.append(tokenize_single(item))
    return list_list_tokens

def token_list_to_ids(list_list_tokens,word2id):
    return [[word2id.get(token,1) for token in list_tokens] for list_tokens in list_list_tokens]

def average_sentence_vectors(list_list_inds,emb_matrix):
    sentence_avgs = []
    for list_inds in list_list_inds:
        sentence_avgs.append(emb_matrix[list_inds].mean(axis=0))
    return np.array(sentence_avgs,dtype='float32')

def filter_labels(df_train, columns):
    return df_train[columns].values.astype('float32')
