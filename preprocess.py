import re,string,collections,pickle,gc,six,math,os,sys,time,datetime,pytz,csv
import chainer,sklearn
from math import log
from time import strftime
from datetime import datetime,timedelta
from pytz import timezone
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


#train_path = '/Users/abhishek/STS/stasis/sts2016_train.stasis.csv'
#test_path = '/Users/abhishek/STS/stasis/sts2016_new.csv'
#glove_path = "/Users/abhishek/Downloads/glove.840B.300d.txt"
def getData(path):
    data = pd.read_csv(path, usecols=['Domain', 'Score','Sent1', 'Sent2'])
    data.dropna(axis=0,how='any',subset=['Domain','Score','Sent1','Sent2'],inplace=True)
    return data
def getTestData(path):
    data = pd.read_csv(path, usecols=['Domain', 'Score','Sent1', 'Sent2'])
    data.dropna(axis=0,how='any',subset=['Domain','Score','Sent1','Sent2'],inplace=True)
    return data

def clean(text):
    return ' '.join([x.strip() for x in re.split('(\W+)?', text) if x.strip()])

def get_all_sentences(data):
    all_sentences = list()
    all_sentences = data.Sent1.tolist() + data.Sent2.tolist()
    return all_sentences


def create_vocab(data):
    frequency_dictionary = {}
    list_of_sentences = get_all_sentences(data)
    list_of_sentences = [clean(sent) for sent in list_of_sentences]
    for each_sentence in list_of_sentences:
        words = each_sentence.split()
        for word in words:
            if word in frequency_dictionary.keys():
                frequency_dictionary[word]+=1.0
            else:
                frequency_dictionary[word]=1.0
    vocab = set(frequency_dictionary.keys())
    return vocab,frequency_dictionary

def get_vectors(vocab,vector_path,out_dir):
    vectors = {}
    with open(vector_path,"r") as f:
        for line in f:
            vals = line.rstrip().split(' ')
            if vals[0] in vocab:
                vectors[vals[0]] =  np.array(vals[1:], ndmin=2, dtype=np.float32)
    vectors['<UNK>'] = np.random.uniform(-0.1, 0.1, (1, 300)).astype(np.float32)
    vectors['<PAD>'] = np.random.uniform(-0.1, 0.1, (1, 300)).astype(np.float32)
    return vectors
def get_vec(word,vectors):
    try:
        vec = vectors[word][0]
    except:
        vec = vectors['<UNK>'][0]
    return vec

def word2vec(sent,sentence_length,vectors):
    words = clean(sent).split()[:sentence_length]
    words = ['<PAD>'] * (sentence_length - len(words)) + words
    return [get_vec(word,vectors) for word in words]

def prepare_sentence_data(data,vectors,sentence_length=32):
    datasets = []
    labels = []
    for i in range(len(data)):
        sent1 = data[i][2]
        sent2 = data[i][3]
        datasets.append([word2vec(sent,sentence_length,vectors) for sent in [sent1,sent2]])
        labels.append(int(round(data[i][1])))
    return datasets,labels
def batch(dataset, indexes):
    return [dataset[i] for i in indexes]
def stack_pairs(sent_batch):
    sents1 = []
    sents2 = []
    for sent1, sent2 in sent_batch:
        sents1.append(sent1)
        sents2.append(sent2)
    return sents1 + sents2


"""
data = getData(train_path)
vocab,frequency_dictionary = create_vocab(data)
get_vectors(vocab,glove_path)
vectors = pickle.load(open("/Users/abhishek/STS/glove_vectors.pkl","r"))
pickle.dump(vocab,open('/Users/abhishek/STS/train_vocab.pkl',"w"))
train_data,validate_data = train_test_split(data,test_size=0.2)
train_data = train_data.as_matrix()
validate_data = validate_data.as_matrix()
train_dataset,train_labels = prepare_sentence_data(train_data)
validation_dataset,validation_labels = prepare_sentence_data(validate_data)
test_dataset,test_labels = prepare_sentence_data(test_data)
"""
