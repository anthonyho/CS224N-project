#!/usr/bin/env python
import pandas as pd
import nltk

def load_data(file_name):
    return pd.read_csv(file_name, encoding='utf-8')

def tokenize_single(sequence):
    tokens = [token.replace("``", '"').replace("''", '"').lower() for token in nltk.word_tokenize(sequence)]
    return tokens

def tokenize_df(df, column='comment_text'):
    list_tokens = []
    for index, item in df[column].iteritems():
        list_tokens.append(tokenize_single(item))
    return list_tokens

def filter_labels(df_train, columns):
    return df_train[columns].values
