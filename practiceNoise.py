import glob
import os
import numpy as np 
from PIL import Image
import cv2
from sklearn.utils import shuffle
from skimage.util import random_noise
import torch
import random
import matplotlib.pyplot as plt
class addNoise():
    def __init__(self):
        print("ay")
        self.IMG_INDEX=30
        self.width=160
        self.height=160
        xtrain,xtest,ytrain,ytest=self.train_test()
        og_img=xtrain[self.IMG_INDEX]
        print(f"og img:{og_img}")
        laplace_img=self.add_laplace_noise(og_img)
        print(laplace_img)
        poisson_img=self.add_poisson_noise(og_img)
        # sp=self.add_sp(og_img,.4)
        # plt.figure()
        f,axarr=plt.subplots(1,3)
        axarr[0].title.set_text("Original")
        axarr[1].title.set_text("LaPlace")
        axarr[2].title.set_text("Poisson")
        axarr[0].imshow(og_img,cmap="gray")
        axarr[1].imshow(laplace_img,cmap="gray")
        axarr[2].imshow(poisson_img,cmap="gray")
        # axarr[3].imshow(sp,cmap="gray")
        plt.show()
    def add_sp(self,images,SNR):
        # Getting the dimensions of the image
        noisy=[]
        for img in images:
            row , col = img.shape
            
            # Randomly pick some pixels in the
            # image for coloring them white
            # Pick a random number between 300 and 10000
            number_of_pixels = random.randint(300, 10000)
            for i in range(number_of_pixels):
                
                # Pick a random y coordinate
                y_coord=random.randint(0, row - 1)
                
                # Pick a random x coordinate
                x_coord=random.randint(0, col - 1)
                
                # Color that pixel to white
                img[y_coord][x_coord] = 255
                
            # Randomly pick some pixels in
            # the image for coloring them black
            # Pick a random number between 300 and 10000
            number_of_pixels = random.randint(300 , 10000)
            for i in range(number_of_pixels):
                
                # Pick a random y coordinate
                y_coord=random.randint(0, row - 1)
                
                # Pick a random x coordinate
                x_coord=random.randint(0, col - 1)
                
                # Color that pixel to black
                img[y_coord][x_coord] = 0
            noisy.append(img)
        return noisy
        # return noisy
    def add_laplace_noise(self,images):
        noisy_images=[]
        for img in images:
            img[img<0]=0
            # img_mask=np.random.laplace(img,scale=1.35)
            new_img=cv2.Laplacian(img,cv2.CV_64FC1)
            # img=img+img_mask
            noisy_images.append(new_img)
        return np.array(noisy_images)
    def add_poisson_noise(self,images):
        noisy_images=[]
        for img in images:
            img[img<0]=0
            for _ in range(4):
                img=random_noise(img,mode="poisson")
            new_img=img.copy()
            # img_mask=np.random.poisson(img)
            # img=img+img_mask
            # new_img=random_noise(img,mode="poisson",var=.005)
            # noise_mask=np.random.poisson(img)
            # new_img = img + noise_mask
            # img_ten=torch.from_numpy(img)
            # img_ten=img_ten.to("cuda:0")
            # new_img=torch.poisson(img_ten)
            noisy_images.append(new_img)
        print(f"{(images == noisy_images[0]).sum(axis=1)}")
        return np.array(noisy_images)
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

c=addNoise()
