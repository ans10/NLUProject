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
import models
from models import NTIFullTreeMatching,BILSTM
import preprocess

def main():
    out_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Get Training path and Glove path')
    parser.add_argument('--gpu', '-g', type=int, default=0,help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--data', '-d', default=out_dir,help='data path')
    parser.add_argument('--glove', '-w', default=out_dir,help='File to read glove vectors')
    parser.add_argument('-t',action='store_true',default=False,dest='testing')
    parser.add_argument('--setvocab','-v',default=None,help="Providing already setup vocab")
    parser.add_argument('--setvectors','-c',default=None)
    parser.add_argument('--model','-m',default='NTIFullTreeMatching')
    args = parser.parse_args()
    gpu = args.gpu
    train_path = os.path.join(args.data,'sts2016_train.stasis.csv')
    glove_path = os.path.join(args.glove,'glove.840B.300d.txt')
    vocab_path = args.setvocab
    vectors_path = args.setvectors
    model_name = args.model
    logging_file = "train.log"
    FORMAT = "%(asctime)-15s %(message)s"
    if(logging_file is None):
        logging.basicConfig(format=FORMAT,level=logging.DEBUG)
    else:
        logging.basicConfig(filename=logging_file,format=FORMAT,level=logging.DEBUG)
    logging.info("The output directory is %s",out_dir)
    logging.info("The glove path is %s",glove_path)
    logging.info("The data path is %s",train_path)
    if(gpu<0):
        logging.info("The program is running for CPU")
    else:
        logging.info("The program is running for GPU")

    n_epoch   = 40   # number of epochs
    n_units   = 300  # number of units per layer
    batch_size = 32  # minibatch size
    eval_batch = 64
    max_dev = 0
    max_tr = 0
    max_test = 0
    max_epch = 0

    EMPTY = np.random.uniform(-0.1, 0.1, (1, 300)).astype(np.float32)
    #preprocessing
    vocab = None
    vectors = None
    data = preprocess.getData(train_path)
    if(vocab_path is not None):
        vocab = pickle.load(open(vocab_path,"r"))
        logging.info("Vocabulary loaded")
    else:
        vocab,frequency_dictionary = preprocess.create_vocab(data)
        pickle.dump(vocab,open(os.path.join(out_dir,"train_vocab.pkl"),"w"))
        logging.info("Dumping the training vocabulary in %s",out_dir+"/train_vocab.pkl")
        logging.info("Vocabulary created")
    if(vectors_path is None):
        vectors = preprocess.get_vectors(vocab,glove_path,out_dir)
        logging.info("Vectors formed")
    else:
        vectors = pickle.load(open(vectors_path,"r"))
        logging.info("Vectors loaded")
    train_data,validate_data = train_test_split(data,test_size=0.1)
    train_data = train_data.as_matrix()
    validate_data = validate_data.as_matrix()
    train_dataset,train_labels = preprocess.prepare_sentence_data(train_data,vectors)
    dataset,labels = preprocess.prepare_sentence_data(validate_data,vectors)
    validation_dataset,validation_labels = dataset,labels
    #test_dataset,test_labels = dataset,labels
    if(args.testing):
        logging.info("Just Testing!")
        train_dataset,train_labels = train_dataset[0:100],train_labels[0:100]
        validation_dataset,validation_labels = dataset[0:100],labels[0:100]
        #test_dataset,test_labels = dataset[-100:],labels[-100:]



    logging.info("The training size is %d",len(train_labels))
    logging.info("The validation size is %d",len(validation_labels))
    #logging.info("The test size is %d",len(test_labels))
    model = None
    if model_name == "BILSTM":
        model = BILSTM(n_units, gpu)
    else:
        model = NTIFullTreeMatching(n_units,gpu)
    model.init_optimizer()
    n_train = len(train_labels)
    n_dev = len(validation_labels)
    #n_test = len(test_labels)
    logging.debug("Training Begins")


    #training code
    for i in xrange(0, n_epoch):
        logging.debug("epoch={}".format(i))
        #Shuffle the data
        shuffle = np.random.permutation(n_train)
        preds=[]
        preds_true=[]
        aLoss = 0
        ss = 0
        begin_time = time.time()
        for j in six.moves.range(0, n_train, batch_size):
            c_b = shuffle[j:min(j+batch_size, n_train)]
            ys = preprocess.batch(train_labels, c_b)
            preds_true.extend(ys)
            y_data = np.array(ys, dtype=np.int32)
            sent_batch = preprocess.batch(train_dataset, c_b)
            sent_batch = preprocess.stack_pairs(sent_batch)
            y_s, loss = model.train(sent_batch, y_data)
            aLoss = aLoss + loss.data
            preds.extend(y_s)
            ss = ss + 1
        logging.debug("loss:%f", aLoss/ss)
        logging.debug('secs per train epoch={}'.format(time.time() - begin_time))
        f1_tr = accuracy_score(preds_true, preds)
        logging.debug('train accuracy_score={}'.format(f1_tr))
        logging.debug(confusion_matrix(preds_true, preds))
        preds = []
        preds_true=[]
        for j in six.moves.range(0, n_dev, eval_batch):
            ys = validation_labels[j:j+eval_batch]
            preds_true.extend(ys)
            y_data = np.array(ys, dtype=np.int32)
            sent_batch =  validation_dataset[j:j+eval_batch]
            sent_batch = preprocess.stack_pairs(sent_batch)
            y_s = model.predict(sent_batch)
            preds.extend(y_s)
        f1_dev = accuracy_score(preds_true, preds)
        logging.debug('dev accuracy_score={}'.format(f1_dev))
        logging.debug(confusion_matrix(preds_true, preds))
        if f1_dev > max_dev:
            max_dev = f1_dev
            max_tr = f1_tr
            max_epch = i
            logging.info('saving model')
            model.save(out_dir + '/'+model_name+'.' + str(i))
        logging.info("best results so far (dev): epoch=%d  dev f1-score=%d  test f1-score=%d",max_epch,max_dev,max_test)
        #if i - max_epch > 5:
        #    logging.warning("No recent improvement on dev, early stopping...")
        #    break
if __name__ == '__main__':
    main()
