import nltk
from collections import Counter
import numpy as np
from os import listdir
import codecs
from random import randint
# import pandas

def sample(data, seq_size):
    x, y = [], []
    idx = randint(0, len(data) - seq_size - 1)
    return np.array(data[idx : idx + seq_size]), np.array(data[idx + 1 : idx + 1 + seq_size])


def get_data(vocab_size):
    f = open('input2.txt','r').read()
    f = f.replace('\n',' eos ')
    id_to_token = {0 : 'unk'}
    words = nltk.word_tokenize(f)
    l = len(set(words))
    print l
    if vocab_size == -1:
        vocab_size = l
    assert vocab_size <= l
    vocab = Counter(words).most_common(vocab_size - 1)
    vocab = [r[0] for r in vocab]
    data = []
    for word in words:
        temp = [0] * (vocab_size)
        if word in vocab:
            temp[vocab.index(word) + 1] = 1
            id_to_token[vocab.index(word) + 1] = word
        else: # unk
            temp[0] = 1
        data.append(temp)
    return data, id_to_token

def hello():
    dirrs = ['sentiment/train/', 'sentiment/test/']
    sent = []
    for dirr in dirrs:
        print dirr
        l = listdir(dirr+'pos')
        print 'pos'
        for r in l:
            t = codecs.open(dirr+'pos/'+r,'r',encoding='utf8').read()
            sent.append(nltk.word_tokenize(t))
        l = listdir(dirr+'neg')
        print 'neg'
        for r in l:
            t = codecs.open(dirr+'neg/'+r,'r',encoding='utf8').read()
            sent.append(nltk.word_tokenize(t))
    return sent
