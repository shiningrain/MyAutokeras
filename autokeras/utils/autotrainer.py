# AutoTrainer: a tool to predict the performance of model;
# If the current structure have potential training problems, it will be stopped by autotrainer
# Future work will add the autofix part.


import os
import sys
import psutil
import csv
import numpy as np
import keras
from keras.models import load_model,Sequential
import keras.backend as K
import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint

from autokeras.utils import monitor as mn
from autokeras.utils import compute_gradient as cg


default_param = {'beta_1': 1e-3,
                 'beta_2': 1e-4,
                 'beta_3': 70,
                 'gamma': 0.7,
                 'zeta': 0.03,
                 'eta': 0.2,
                 'delta': 0.01,
                 'alpha_1': 0,
                 'alpha_2': 0,
                 'alpha_3': 0,
                 'Theta': 0.55
                 }

def read_data(dataset):
    # read data and batchsize from a new unzipped dataset.
    trainX=dataset[0][0]
    batch_size=max(dataset[0][1].shape)
    trainy=dataset[0][1].reshape(batch_size,)
    return trainX,trainy,batch_size


class LossHistory(keras.callbacks.Callback):

    def __init__(self,training_data,model,total_epoch,determine_threshold=3,satisfied_acc=0.7,\
        checktype='epoch_1',params={}): #only support epoch method now
        """[summary]

        Args:
            training_data ([list]): [training dataset]
            model ([model]): [untrained model]
            batch_size ([int]): [batch size]
            save-dir([str]):[the dir to save the detect result]
            checktype (str, optional): [checktype,'a_b', a can be chosen from ['epoch', 'batch'], b is number, it means the monitor will check \
            the gradient and loss every 'b' 'a'.]. Defaults to 'epoch_5'.
            satisfied_acc (float, optional): [the satisfied accuracy, when val accuracy beyond this, the count ++, when the count is bigger or\
                equal to satisfied_count, training stop.]. Defaults to 0.7.

        """
        self.trainX,self.trainy,self.batch_size = read_data(training_data)
        self.model=model
        self.satisfied_acc=satisfied_acc
        self.count=0
        self.checktype=checktype.split('_')[0]
        self.checkgap=int(checktype.split('_')[-1])
        self.issue_list=[]
        self.total_epoch=total_epoch
        self.determine_threshold=determine_threshold
        self.params=params
        if self.params=={}:
            self.params=default_param

        self.history={}
        self.history['loss']=[]
        self.history['acc']=[]
        self.history['val_loss']=[]
        self.history['val_acc']=[]

        self.Monitor=mn.IssueMonitor(total_epoch,self.satisfied_acc,self.params,self.determine_threshold)

    def on_train_begin(self,logs=None):
        weights=self.model.trainable_weights# get trainable weights
        if not cg.check_version(tf.__version__):
            try:
                grads = self.model.optimizer.get_gradients(self.model.total_loss, weights)
                symb_inputs = [self.model._feed_inputs , self.model._feed_targets , self.model._feed_sample_weights,K.learning_phase()]#input,corresponding label,weight of each sample(all of them are 1),learning rate(we set it to 0)
                self.f = K.function(symb_inputs, grads)
                self.new_computation=False
            except:
                self.new_computation=True
        else:
            self.new_computation=True       


    def on_epoch_end(self,epoch,logs={}):
        self.history['loss'].append(logs.get('loss'))
        self.history['acc'].append(logs.get('accuracy'))
        self.history['val_loss'].append(logs.get('val_loss'))
        self.history['val_acc'].append(logs.get('val_accuracy'))
        if (epoch)%self.checkgap==0:
            
            trainingExample = self.trainX
            trainingY=self.trainy
            if self.new_computation==False:
                x, y, sample_weight = self.model._standardize_user_data(trainingExample, trainingY)
                #output_grad = f(x + y + sample_weight)
                self.evaluated_gradients = self.f([x , y , sample_weight,0])
            else:
                self.evaluated_gradients = cg.get_gradients(self.model, trainingExample, trainingY)
            gradient_list=[]
            for i in range(len(self.evaluated_gradients)):
                if isinstance(self.evaluated_gradients[i],np.ndarray):
                    gradient_list.append(self.evaluated_gradients[i])

            self.issue_list=self.Monitor.determine(self.model,self.history,gradient_list,self.checkgap)

            self.evaluated_gradients=0
            gradient_list=0

            if self.issue_list!=[]:
                self.issue_list=list(set(self.issue_list))
                self.model.stop_training = True
                print('\nThis Model have potential training problem:',self.issue_list)
                print('Stop current training and search for next model')


    def on_train_end(self,logs=None):
        print('Finished Training')