import joblib
import numpy as np
from PIL import Image
from keras.models import load_model
import keras
import os
import hashlib


print('agrg?')
class FlowerGen():

    def __init__(self):
        self.generator = load_model('./models/saved_models/DCGAN_V1.h5')
        self.clf = joblib.load('./models/saved_models/refiner_SVC.joblib')
        self.noise = np.random.normal(loc=0,scale=1,size=(1,100))
        self.lasthash = None

        self.get_new_noise()
        self.gen_new_image()

    def get_new_noise(self):
        self.noise = np.random.normal(loc=0,scale=1,size=(1,100))
        while self.clf.predict_proba(self.noise)[0,1] < 0.5:
            self.noise = np.random.normal(loc=0,scale=1,size=(1,100))
        
    def gen_new_image(self):
        img = self.generator.predict(self.noise)[0] * 0.5 + 0.5
        image = Image.fromarray((img*255).astype('uint8'))
        md5hash = hashlib.md5(image.tobytes()).hexdigest()
        image.save(f'./static/images/{md5hash}.png')
        if self.lasthash is not None:
            os.remove(f'./static/images/{self.lasthash}.png')
        else:
            pass
        self.lasthash = md5hash
        self.get_new_noise()
        return f'images/{self.lasthash}.png'
        

if __name__ == '__main__':
    flowergen = FlowerGen()
    flowergen.gen_new_image()
    flowergen.gen_new_image()
