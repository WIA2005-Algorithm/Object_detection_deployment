from flask import Flask, request,render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np 
from PIL import Image
app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict' ,methods=['POST','GET'])

def function():
    class_names= ['Non_recyclable', 'Recyclable']
    img_height = 180
    img_width = 180
    model = tf.keras.models.load_model('saved_model\my_model')
    data= request.files['data']
    image = Image.open(data)
    image = image.resize((img_height, img_width))

    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    classe_name = class_names[np.argmax(score)]
    score = round(100 * np.max(score),2)

    
    return render_template('index.html', score=score, classe_name=classe_name)