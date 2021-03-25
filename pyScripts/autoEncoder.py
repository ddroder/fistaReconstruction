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
class Autoencoder():
    def __init__(self):
        self.img_rows = 160
        self.img_cols = 160
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        optimizer = Adam(lr=0.001)
        
        self.autoencoder_model = self.build_model()
        self.autoencoder_model.compile(loss='mse', optimizer=optimizer)
        self.autoencoder_model.summary()
    
    def build_model(self):
        """
        This is the model that will 
        be used as a convolutional autoencoder.
        """
        input_layer = Input(shape=self.img_shape)
        
        # encoder
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
        h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
        h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
        h = MaxPooling2D((2, 2), padding='same')(h)
        
        # decoder
        h = Conv2D(64, (3, 3), activation='relu', padding='same')(h)
        h = Conv2D(128, (3, 3), activation='relu', padding='same')(h)
        h = Conv2D(256, (3, 3), activation='relu', padding='same')(h)
        h = Conv2D(512, (3, 3), activation='relu', padding='same')(h)
        h = UpSampling2D((2, 2))(h)
        output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(h)
        
        return Model(input_layer, output_layer)
    # def psnr(self,y_actual,y_hat):
        
    def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size=20,poisson=False,laplace=False,fista=False,saveModel=False):
        """
        this is a method that is the training loop.
        """
        if laplace:
            x_train=self.add_laplace_noise(x_train)
            x_val=self.add_laplace_noise(x_val)
        elif poisson:
            x_train=self.add_poisson_noise(x_train)
            x_val=self.add_poisson_noise(x_val)
        elif fista:
            x_train=self.fista_encode_images(x_train)
            x_val=self.fista_encode_images(x_val)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=7,
                                       verbose=1, 
                                       mode='auto')
        history = self.autoencoder_model.fit(x_train, y_train,
                                             batch_size=batch_size,
                                             epochs=epochs,
                                             validation_data=(x_val, y_val),
                                             callbacks=[early_stopping])
        if saveModel:
            self.saveModel(self.autoencoder_model)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
    def encode_images(self,images,poisson=False,laplace=False):
        if poisson:
            images=self.add_poisson_noise(images)
        if laplace:
            images=self.add_laplace_noise(images)
        encoded_images=self.fista_encode_images(images)
        return encoded_images
    def add_poisson_noise(self,images):
        noisy_images=[]
        for img in images:
            img[img<0]=0
            img_mask=np.random.poisson(img/255.0 * .05)/.05*255
            img=img+img_mask
            noisy_images.append(img)
        return np.array(noisy_images)
    def add_laplace_noise(self,images):
        noisy_images=[]
        for img in images:
            img[img<0]=0
            img_mask=np.random.laplace(img,scale=.05)
            img=img+img_mask
            noisy_images.append(img)
        return np.array(noisy_images)
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
            img_encoded=img_encoded.reshape(self.img_rows,self.img_cols)
            img_encoded=img_encoded.detach().cpu().numpy()
            encoded_images.append(img_encoded)
        return np.array(encoded_images)
    def eval_model(self, x_test):
        preds = self.autoencoder_model.predict(x_test)
        return preds
    def saveModel(self,model):
        model.save("autoEncoder.h5")
        print("Model saved!")