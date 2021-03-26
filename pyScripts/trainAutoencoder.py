import glob
import numpy as np
from tensorflow.keras.preprocessing import image
from autoEncoder import Autoencoder

Yes_IMAGES = glob.glob('/nvme_ssd/bensCode/kaggleMRI/brain_tumor_dataset/yes/*.jpg')
No_IMAGES = glob.glob('/nvme_ssd/bensCode/kaggleMRI/brain_tumor_dataset/no/*.jpg')
def load_image(path):
    image_list = np.zeros((len(path), 160, 160, 1))
    for i, fig in enumerate(path):
        img = image.load_img(fig, color_mode='grayscale', target_size=(160, 160))
        x = image.img_to_array(img).astype('float32')
        x = x / 255.0
        image_list[i] = x
    
    return image_list
x_train = load_image(Yes_IMAGES)
y_train = load_image(Yes_IMAGES)
x_test = load_image(No_IMAGES)
def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]
x_train, y_train, x_val, y_val = train_val_split(x_train, y_train)
ae = Autoencoder()
ae.train_model(x_train, y_train, x_val, y_val, epochs=150, batch_size=20,laplace=True,saveModel=False)