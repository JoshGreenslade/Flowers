import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from flask import Flask, render_template
from models.flower_generator import FlowerGen

app = Flask(__name__)

flowergen = FlowerGen()

@app.route('/')
def home():
    img = flowergen.gen_new_image()
    return render_template('home.html', img=img)

if __name__ == '__main__':
    app.run(debug=True)