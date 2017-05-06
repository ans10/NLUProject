import re,string,collections,pickle,gc,six,math,os,sys,time,datetime,pytz,csv
import chainer,sklearn
from math import log
import pandas as pd
import numpy as np
import scipy as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


class NTIFullTreeMatching(chainer.Chain):

    """docstring for NTIFullTreeMatching"""
    def __init__(self, n_units, gpu,sentence_length=32):
        super(NTIFullTreeMatching, self).__init__(
            h_lstm = L.LSTM(n_units, n_units),
            m_lstm = L.LSTM(n_units, n_units),
            h_x = F.Linear(n_units, 4*n_units),
            h_h = F.Linear(n_units, 4*n_units),
            w_ap = F.Linear(n_units, n_units),
            w_we = F.Linear(n_units, 1),
            w_c = F.Linear(n_units, n_units),
            w_q = F.Linear(n_units, n_units),
            h_l1 = F.Linear(2*n_units, 1024),
            l_y = F.Linear(1024, 6))
        self.__n_units = n_units
        self.__sentence_length = sentence_length
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        for param in self.params():
            data = param.data
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def init_optimizer(self):
        self.__opt = optimizers.Adam(alpha=0.00003, beta1=0.9, beta2=0.999, eps=1e-08)
        self.__opt.setup(self)
        self.__opt.add_hook(chainer.optimizer.GradientClipping(40))
        self.__opt.add_hook(chainer.optimizer.WeightDecay(0.00003))

    def save(self, filename):
        chainer.serializers.save_npz(filename, self)

    @staticmethod
    def load(filename, n_units, gpu):
        self = NTIFullTreeMatching(n_units, gpu)
        chainer.serializers.load_npz(filename, self)
        return self

    def reset_state(self):
        self.h_lstm.reset_state()
        self.m_lstm.reset_state()

    def __attend_f_tree(self, hs, hsq, q, batch_size, train):
        n_units = self.__n_units
        mod = self.__mod

        # calculate attention weights
        x_len = len(hs)
        depth = int(log(x_len, 2)) + 1

        w_a = F.reshape(F.batch_matmul(F.dropout(hsq, ratio=0.0, train=train), self.w_ap(q)), (batch_size, -1))
        w_a = F.exp(w_a)
        list_e = F.split_axis(w_a, x_len, axis=1)

        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 2):
                l = hs[s]
                r = hs[s+1]
                lr = hs[(s-1)/2]
                le = list_e[s]
                re = list_e[s+1]
                lre = list_e[(s-1)/2]
                sum_e = le + re + lre
                lr = F.batch_matmul(lr, lre/sum_e)
                lr += F.batch_matmul(l, le/sum_e)
                lr += F.batch_matmul(r, re/sum_e)
                hs[(s-1)/2] = F.reshape(lr, (batch_size, -1))


        s_c = hs[0]
        s_c = F.relu(self.w_c(s_c) + self.w_q(q))

        return s_c

    def __attend_fast(self, hs, q, batch_size, train):
        n_units = self.__n_units
        mod = self.__mod

        w_a = F.reshape(F.batch_matmul(F.dropout(hs, ratio=0.0, train=train), self.w_ap(q)), (batch_size, -1))
        w_a = F.softmax(w_a)
        s_c = F.reshape(F.batch_matmul(w_a, hs, transa=True), (batch_size, -1))

        h = F.relu(self.w_c(s_c) + self.w_q(q))
        return h


    def __forward(self, train, x_batch, y_batch = None):
        model = self
        n_units = self.__n_units
        mod = self.__mod
        gpu = self.__gpu
        batch_size = len(x_batch)
        x_len = len(x_batch[0])
        depth = int(log(x_len, 2)) + 1

        self.reset_state()

        list_a = [[] for i in range(2**depth-1)]
        list_c = [[] for i in range(2**depth-1)]
        zeros = mod.zeros((batch_size, n_units), dtype=np.float32)
        for l in xrange(x_len):
            x_data = mod.array([x_batch[k][l] for k in range(batch_size)])
            x_data = Variable(x_data, volatile=not train)
            x_data = model.h_lstm(F.dropout(x_data, ratio=0.2, train=train))
            list_a[x_len-1+l] = x_data
            list_c[x_len-1+l] = model.h_lstm.c #Variable(zeros, volatile=not train)

        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 2):
                l = model.h_x(F.dropout(list_a[s], ratio=0.2, train=train))
                r = model.h_h(F.dropout(list_a[s+1], ratio=0.2, train=train))
                c_l = list_c[s]
                c_r = list_c[s+1]
                c, h = F.slstm(c_l, c_r, l, r)
                list_a[(s-1)/2] = h
                list_c[(s-1)/2] = c

        list_p = []
        list_h = []
        for a in list_a:
            n_hs = F.split_axis(a, 2, axis=0)
            list_p.append(n_hs[0])
            list_h.append(n_hs[1])

        list_pq = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_p], axis=1)
        list_aoa = []
        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 1):
                a = self.__attend_fast(list_pq, list_h[s], batch_size/2, train)
                # a = self.__attend_f_tree(list_p[:], list_pq, list_h[s], batch_size/2, train)
                hs = model.m_lstm(F.dropout(a, ratio=0.2, train=train))
                list_aoa.append(hs)
        list_aoa = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_aoa[:-1]], axis=1)
        hs = self.__attend_fast(list_aoa, hs, batch_size/2, train)

        model.m_lstm.reset_state()

        list_pq = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_h], axis=1)
        list_aoa = []
        for d in reversed(range(1, depth)):
            for s in range(2**d-1, 2**(d+1)-1, 1):
                a = self.__attend_fast(list_pq, list_p[s], batch_size/2, train)
                # a = self.__attend_f_tree(list_h[:], list_pq, list_p[s], batch_size/2, train)
                hs1 = model.m_lstm(F.dropout(a, ratio=0.2, train=train))
                list_aoa.append(hs1)
        list_aoa = F.concat([F.reshape(h, (batch_size/2, 1, n_units)) for h in list_aoa[:-1]], axis=1)
        hs1 = self.__attend_fast(list_aoa, hs1, batch_size/2, train)

        hs = F.relu(model.h_l1(F.concat([hs, hs1], axis=1)))
        y = model.l_y(F.dropout(hs, ratio=0.2, train=train))
        preds = mod.argmax(y.data, 1).tolist()

        accum_loss = 0 if train else None
        if train:
            if gpu >= 0:
                y_batch = cuda.to_gpu(y_batch)
            lbl = Variable(y_batch, volatile=not train)
            accum_loss = F.softmax_cross_entropy(y, lbl)

        return preds, accum_loss

    def train(self, x_batch, y_batch):
        self.__opt.zero_grads()
        preds, accum_loss = self.__forward(True, x_batch, y_batch=y_batch)
        accum_loss.backward()
        self.__opt.update()
        return preds, accum_loss

    def predict(self, x_batch):
        return self.__forward(False, x_batch)[0]


class BILSTM(chainer.Chain):

    def __init__(self, n_units, gpu,sentence_length=32,method='concat'):
        super(BILSTM, self).__init__(
            fwd_lstm_s1 = L.LSTM(n_units, n_units),
            bwd_lstm_s1 = L.LSTM(n_units, n_units),
            fwd_lstm_s2 = L.LSTM(n_units,n_units),
            bwd_lstm_s2 = L.LSTM(n_units,n_units),
            bi_output_layer = L.Linear(2*n_units,1),
            output_layer = L.Linear(2*sentence_length,6))
        self.__n_units = n_units
        self.__gpu = gpu
        self.__mod = cuda.cupy if gpu >= 0 else np
        for param in self.params():
            data = param.data
            data[:] = np.random.uniform(-0.1, 0.1, data.shape)
        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()

    def init_optimizer(self):
        self.__opt = optimizers.Adam(alpha=0.0003, beta1=0.9, beta2=0.999, eps=1e-08)
        self.__opt.setup(self)
        #self.__opt.add_hook(chainer.optimizer.GradientClipping(40))
        #self.__opt.add_hook(chainer.optimizer.WeightDecay(0.00003))

    def save(self, filename):
        chainer.serializers.save_npz(filename, self)

    @staticmethod
    def load(filename, n_units, gpu):
        self = BILSTM(n_units, gpu)
        chainer.serializers.load_npz(filename, self)
        return self

    def reset_state(self):
        self.fwd_lstm_s1.reset_state()
        self.bwd_lstm_s1.reset_state()
        self.fwd_lstm_s2.reset_state()
        self.bwd_lstm_s2.reset_state()



    def __forward(self,train,x_batch,y_batch=None):
        model = self
        n_units = self.__n_units
        mod = self.__mod
        gpu = self.__gpu
        batch_size = len(x_batch)/2
        sentence_length = len(x_batch[0])
        #print batch_size
        model.reset_state()
        x_batch = np.array(x_batch)
        final_output=None


        for l in range(len(x_batch[0])):
            x_data = mod.array([x_batch[k][l] for k in range(batch_size)])
            x_data = Variable(x_data, volatile=not train)
            x_data = model.fwd_lstm_s1(x_data)

            bwd_x_data = mod.array([x_batch[k][len(x_batch[0])-l-1] for k in range(batch_size)])
            bwd_x_data = Variable(bwd_x_data,volatile=not train)
            bwd_x_data = model.bwd_lstm_s1(bwd_x_data)

            x_data_2 = mod.array([x_batch[k][l] for k in range(batch_size,2*batch_size)])
            x_data_2 = Variable(x_data_2, volatile=not train)
            x_data_2 = model.fwd_lstm_s2(x_data_2)

            bwd_x_data_2 = mod.array([x_batch[k][len(x_batch[0])-l-1] for k in range(batch_size,2*batch_size)])
            bwd_x_data_2 = Variable(bwd_x_data_2,volatile=not train)
            bwd_x_data_2 = model.bwd_lstm_s2(bwd_x_data_2)


            s1_output = model.bi_output_layer(F.concat((x_data,bwd_x_data),axis=1))
            s2_output = model.bi_output_layer(F.concat((x_data_2,bwd_x_data_2),axis=1))
            #print s1_output.shape
            #print s2_output.shape
            #subtracting
            #elementwise multiplication
            if(method=='concat'):
                output = F.concat((s1_output,s2_output),axis=1)
            elif(method=='difference'):
                output = s1_output - s2_output
            #print output.shape
            if(l==0):
                final_output = output
            else:
                final_output = F.concat((final_output,output),axis=1)

        #print final_output.shape
        y = model.output_layer(final_output)
        preds = mod.argmax(y.data, 1).tolist()
        #print len(preds)
            #print sent2_batch.shape
            #print type(sent1_batch)

        accum_loss = 0 if train else None
        if train:
            if gpu >= 0:
                y_batch = cuda.to_gpu(y_batch)
            lbl = Variable(y_batch, volatile=not train)
            accum_loss = F.softmax_cross_entropy(y, lbl)



        return preds,accum_loss







    def train(self, x_batch, y_batch):
        self.__opt.zero_grads()
        preds, accum_loss = self.__forward(True, x_batch, y_batch=y_batch)
        accum_loss.backward()
        self.__opt.update()
        return preds, accum_loss

    def predict(self, x_batch):
        return self.__forward(False, x_batch)[0]
