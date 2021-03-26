import datetime
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np 
import pandas as pd 
import torch.nn.functional as F
import torch.nn as nn
import os
from os import listdir
import tensorflow as tf
import torch
from keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
import imutils  
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
import tensorflow as tf
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Conv2D,Input,ZeroPadding2D,BatchNormalization,Flatten,Activation,Dense,MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.preprocessing import image
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
import pickle
from keras.models import Sequential
import numpy as np 
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from PIL import Image
from keras.datasets import mnist
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import sys

#python files
sys.path.insert(1, '/nvme_ssd/testing/fistaReconstruction/UTILS')
from class_dict import dictionary
sys.path.insert(1, '/nvme_ssd/testing/fistaReconstruction')
from loadImDat import scaleTo01,loadData

#Create classifier class
class classifier():
    def __init__(self):
        self.width=160
        self.height=160
        pass
    def get_fista_encoder(self):
        model_dir="/nvme_ssd/bensCode/SparseCoding/models/mriModelKaggle.pt"
        fista_model=torch.load(model_dir)
        return fista_model
    def fista_encode_images(self,images):
        #assumption is images is an array of numpy images
        # model=self.get_fista_encoder()
        fista_model=self.get_fista_encoder()
        encoded_images=[]
        for img in images:
            img=torch.tensor(img).to("cuda:0")
            img=img.float()
            # print(img.shape)
            img=img.reshape(1,-1)
            img_encoded=fista_model(img)
            img_encoded=img_encoded.reshape(self.width,self.height)
            img_encoded=img_encoded.detach().cpu().numpy()
            encoded_images.append(img_encoded)
        return np.array(encoded_images)
    def train_test(self,laplace=False,saltPepper=False,fista=False):
        files=glob.glob("/nvme_ssd/bensCode/kaggleMRI/brain_tumor_dataset/**/*.jpg")
        # print(files)
        labels=[]
        for file in files:
            if "no" in file:
                labels.append(0)
            else:
                labels.append(1)
        self.images=[]
        for file in files:
            img=Image.open(file)
            image=img.resize((self.width,self.height),Image.ANTIALIAS)
            array_img=np.array(image)
            arr_img=array_img[:,:,0] if len(array_img.shape)==3 else array_img
            self.images.append(arr_img)
        images,labels=shuffle(self.images,labels)
        arrimg=np.array(images)
        # print(arrimg)
        length=len(labels)
        size=round(length*.8)
        xtrain=arrimg[:size,:self.width,:self.height,np.newaxis]
        xtest=arrimg[:size,:self.width,:self.height,np.newaxis]
        ytrain=labels[:size]
        ytest=labels[:size]
        if fista:
            xtrain=self.fista_encode_images(xtrain)
            xtrain=xtrain.reshape(-1,160,160,1)
            xtest=self.fista_encode_images(xtest)
            xtest=xtest.reshape(-1,160,160,1)
        return xtrain,xtest,ytrain,ytest
    
    def classifierModel(self):
        model = Sequential()
        model.add(Conv2D(50, kernel_size=(3,3), padding='same', activation='relu', input_shape=(160, 160,1)))
        model.add(Conv2D(75, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(125, kernel_size=(3,3), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(250, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
        return model
    def trainModel(self,epochs,xtrain,xtest,ytrain,ytest,saveModel=False):
        model=self.classifierModel()
        log_dir = "tbLogs/Classifier" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        history=model.fit(xtrain,np.array(ytrain),
                            epochs=epochs,
                            validation_data=(xtest,np.array(ytest)),
                            callbacks=[tensorboard_callback])
        if saveModel:
            self.saveModel(model)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    def saveModel(self,model):
        # stringSave=f"{saveDir}/{name}"
        model.save("mriClassifier.h5")
        print("model saved!")

#begin model creation and training
