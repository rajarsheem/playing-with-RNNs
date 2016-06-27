# helper functions for all notebooks

import nltk
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import numpy as np
from os import listdir
import codecs
from random import randint


def sample(data, seq_size):
    idx = randint(0, len(data) - seq_size - 1)
    return np.array(data[idx: idx + seq_size]), np.array(data[idx + 1: idx + 1 + seq_size])


def get_data(vocab_size):
    tzr = RegexpTokenizer(r'\w+')
    f = codecs.open('input2.txt', 'r', encoding='utf8').read()
    # f = f.replace('\n', ' eos ')
    id_to_token = {0: 'unk'}
    words = nltk.word_tokenize(f)
    words = list(map(lambda x: x.lower(), words))
    print(words[:40])
    l = len(set(words))
    print(l)
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
        else:  # unk
            temp[0] = 1
        data.append(temp)
    return data, id_to_token


def hello():
    dirrs = ['sentiment/train/', 'sentiment/test/']
    sent = []
    for dirr in dirrs:
        print(dirr)
        l = listdir(dirr + 'pos')
        print('pos')
        for r in l:
            t = codecs.open(dirr + 'pos/' + r, 'r', encoding='utf8').read()
            sent.append(nltk.word_tokenize(t))
        l = listdir(dirr + 'neg')
        print('neg')
        for r in l:
            t = codecs.open(dirr + 'neg/' + r, 'r', encoding='utf8').read()
            sent.append(nltk.word_tokenize(t))
    return sent


# currently using one-hot.
def get_char_embedding():
    f = codecs.open('input2.txt', 'r', encoding='utf-8').read()
    chars = set(f)
    chars = [i.lower() for i in chars]
    print(chars)
    mat = np.zeros([len(chars), len(chars)])
    np.fill_diagonal(mat, 1)
    char_to_id = {k: v for v, k in enumerate(chars)}
    return char_to_id, mat


def readseq(name):
    idd, seq = [], []
    with open(name) as f:
        for row in f:
            if 'Id' in row:
                continue
            r = row.split('"')
            idd.append(int(r[0][:-1]))
            seq.append(list(map(int, r[1].split(','))))
    return idd, seq

get_char_embedding()
