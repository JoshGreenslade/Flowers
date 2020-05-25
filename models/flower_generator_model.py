""" For posterity only!
This code is not meant to be run, but as a reminder of what I did.
If you do want to run it, you should be able to replicate roughly what I did using jupyter notebooks
"""


from __future__ import print_function, division
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
import keras.layers as layers
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
import numpy as np


class DCGAN():
    """A DCGAN for generating images of flowers.
    """
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.data_dir = './data_paintings/'

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def get_batch(self):
        imgs = next(self.data_generator)[0]
        imgs = imgs/127.5 - 1
        return imgs
    
    def reverse_norm(self, imgs):
        imgs = (imgs + 1)/2.
        return imgs
    
    
    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 4 * 4, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((4, 4, 128)))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(256, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(128, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(64, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(32, kernel_size=3, padding='same'))
        model.add(layers.BatchNormalization(momentum=0.8))
        model.add(layers.LeakyReLU(0.2))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(layers.Conv2D(64, 
                        kernel_size=(3,3), 
                        strides=(2,2), 
                        padding='same',
                        input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(0.2))
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(128, 
                        kernel_size=(3,3), 
                        strides=(2,2), 
                        padding='same',
                        input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.25))
        model.add(layers.Conv2D(256, 
                        kernel_size=(3,3), 
                        strides=(1,1), 
                        padding='same',
                        input_shape=self.img_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        
        # Data directory
        data_generator = ImageDataGenerator(horizontal_flip=True)
        self.data_generator = data_generator.flow_from_directory(self.data_dir,
                                                                 target_size=self.img_shape[:2],
                                                                 batch_size=batch_size,
                                                                 class_mode='input',
                                                                 shuffle=True)
    

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            # Select a random half of images
            imgs = self.get_batch()
            sample_size = len(imgs)
            
            # Adversarial ground truths
            valid = np.ones((sample_size, 1))
            fake = np.zeros((sample_size, 1))

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (sample_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss), end='\r')

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
            if epoch % 5000 == 0:
                self.generator.save(f'Model_Bob_{epoch}')

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c, figsize=(25,25))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        plt.tight_layout()
        fig.savefig("images/bobross_%s.png" % epoch)
        plt.close()



# This next section of code was how I manually trained an SVC to show only 'good' images.
# All run in jupyter
import ipywidgets as widgets
from IPython.display import clear_output
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np

button_yes = widgets.Button(description='Yes')
button_no = widgets.Button(description='No')
out = widgets.Output()
global noise
global codes
codes = np.loadtxt('codes100yes.csv', delimiter=',')

def disp_next_img():
    clear_output(wait=True)
    noise = np.random.normal(loc=0,scale=1,size=(1,101))
    img = gen2.predict(noise[:,:100])[0] * 0.5 + 0.5
    plt.imshow(img)
    plt.gca().axis('off')
    plt.show()
    return noise

def yes_button_clicked(_, result=1):
    global noise
    global codes
    with out:
        noise[0][-1] = result
        codes = np.vstack([codes, noise])
        noise = disp_next_img()
        return noise
output = button_yes.on_click(yes_button_clicked)

def no_button_clicked(_):
    return yes_button_clicked(_, result=0)
button_no.on_click(no_button_clicked)

disp_next_img()
buttons = widgets.HBox([button_yes, button_no])
widgets.VBox([buttons, out])


codes = np.loadtxt('codes150yes.csv', delimiter=',')
clf = SVC(probability=True)
clf.fit(codes[:,:100], codes[:,-1])
dump(clf, 'models/refiner_SVC.joblib') 