import numpy as np
import pickle

from stop_word import remove_stopwords
from prediction import predict
from padded_seq import seq_and_pad


def model(sentence, tokenizer, PADDING, MAXLEN, path):
    sentence = remove_stopwords(sentence)

    with open(tokenizer, 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    padded_seq = seq_and_pad(tokenizer,[sentence], PADDING, MAXLEN)

    out = predict(path, padded_seq)
    return out
