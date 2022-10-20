import json
import os
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse

model = None

def init():
    logging.info("Init started")
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model')
    print(model_path)
    model = tf.keras.models.load_model(model_path)
    logging.info("Init completed")

@rawhttp
def run(request):
    logging.info("Run started")
    class_names = ['Non_recyclable', 'Recyclable']
    img_height = 180
    img_width = 180

    if request.method == 'GET':
        respBody = str.encode(request.full_path)

        logging.info("Run completed")
        return AMLResponse(respBody, 200)
    elif request.method == 'POST':
        file_bytes = request.files["image"]
        image = Image.open(file_bytes).convert('RGB')
        image = image.resize((img_height, img_width))

        img_array = tf.keras.utils.img_to_array(image)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        classe_name = class_names[np.argmax(score)]
        score = round(100 * np.max(score),2)

        logging.info("Run completed")
        return AMLResponse(json.dumps({
            "image_size": image.size,
            "score": score,
        }), 200)
    else:
        logging.info("Run completed")
        return AMLResponse("bad request", 500)
