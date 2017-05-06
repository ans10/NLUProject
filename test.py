import re,string,collections,pickle,gc,six,math,os,sys,csv,time
import chainer,sklearn
import argparse
from math import log
import pandas as pd
import numpy as np
import scipy as sp
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
import preprocess
import models
from models import NTIFullTreeMatching


def main():
    out_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Get Training path and Glove path')
    parser.add_argument("--model","-m",default=os.path.join(out_dir,"NTIFullTreeMatching.0"))
    parser.add_argument("--data","-d",default=os.path.join(out_dir,"sts2016_new.csv"))
    parser.add_argument('--setvocab','-v',default=None,help="Providing already setup vocab")
    parser.add_argument('--setvectors','-c',default=None,help="Providing already setup vectors")
    parser.add_argument('--gpu', '-g', type=int, default=-1,help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--glove', '-w', default=out_dir,help='File to read glove vectors')
    n_units = 300
    args = parser.parse_args()
    model_name = args.model
    test_path = args.data
    vocab_path = args.setvocab
    vectors_path = args.setvectors
    gpu = args.gpu
    glove_path = os.path.join(args.glove,'glove.840B.300d.txt')
    logging_file = "test.log"
    FORMAT = "%(asctime)-15s %(message)s"
    if(logging_file is None):
        logging.basicConfig(format=FORMAT,level=logging.DEBUG)
    else:
        logging.basicConfig(filename=logging_file,format=FORMAT,level=logging.DEBUG)

    data = preprocess.getTestData(test_path)
    if(vocab_path is not None):
        vocab = pickle.load(open(vocab_path,"r"))
        logging.info("Vocabulary loaded")
    else:
        vocab,frequency_dictionary = preprocess.create_vocab(data)
        pickle.dump(vocab,open(os.path.join(out_dir,"test_vocab.pkl"),"w"))
        logging.info("Dumping the testing vocabulary in %s",os.path.join(out_dir,"test_vocab.pkl"))
        logging.info("Vocabulary created")
    if(vectors_path is None):
        vectors = preprocess.get_vectors(vocab,glove_path,out_dir)
        pickle.dump(vectors,open(os.path.join(out_dir,"glove_vectors_test.pkl"),"w"))
        logging.info("Vectors formed")
    else:
        vectors = pickle.load(open(vectors_path,"r"))
        logging.info("Vectors loaded")
    test_data = data.as_matrix()
    test_dataset,test_labels = preprocess.prepare_sentence_data(test_data,vectors)

    #Loading model
    logging.debug("Loading Model: %s",model_name)
    model = NTIFullTreeMatching.load(model_name, n_units, gpu)
    logging.debug("Model loaded")
    preds = []
    sent_batch = test_dataset
    sent_batch = preprocess.stack_pairs(sent_batch)
    y_s = model.predict(sent_batch)
    preds.extend(y_s)
    f1_test = accuracy_score(test_labels, preds)
    logging.debug('test accuracy_score={}'.format(f1_test))
    logging.debug(confusion_matrix(test_labels, preds))
    false_predictions = []
    fout = open("false_predictions.csv","w")
    for i in range(len(test_labels)):
        if(test_labels[i]!=preds[i]):
            outline = '\t'.join([test_data[i][0],test_data[i][2],test_data[i][3],str(test_data[i][1]),str(preds[i])]).strip()
            fout.write(outline+'\n')
    fout.close()





if __name__ == '__main__':
    main()
