import sys
import keras
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
from util import base64_to_pil

# Declare a flask app
app = Flask(__name__)

# Model saved with model.save()
MODEL_PATH = './models/skinmodel.h5'
# MODEL_PATH2 = './models/hairmodel.h5'
# trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
train = pd.read_csv('./models/skin_upd.csv')

# model2 = load_model(MODEL_PATH2)
# model2._make_predict_function()
# train2= pd.read_csv('./models/hair_dataset.csv')

def model_predict(img, model):

    # img = image.load_img(img,target_size=(400,400,3))
    img = img.resize((400, 400))
    img = image.img_to_array(img)
    img = img/255
    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    proba  = model.predict(img.reshape(1,400,400,3)) 
    return proba

# def model_predict2(img, model2):
#     img = img.resize((400, 400))
#     img = image.img_to_array(img)
#     img = img/255
#     proba2  = model2.predict(img.reshape(1,400,400,3))
#     return proba2


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        # preds2= model_predict2(img, model2)
        classes = np.array(train.columns[2:])
        top_3 = np.argsort(preds[0])[:-5:-1]
        # top_3 = np.argsort(preds2[0])[:-5:-1]
        top_3[0]=top_3[0]-1
        top_3[1]=top_3[1]-1
        top_3[2]=top_3[2]-1
        top_3[3]=top_3[3]-1
        # print(top_3)
        # for i in range(4):
        #     print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
        # plt.imshow(img)
        
        
        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        print(pred_proba)
        # pred_class = classes(preds, top=1)   # ImageNet Decode
        result1 = []
        result2 = {}
        for i in range(4):
            result = str(classes[top_3[i]])          # Convert to string
            result = result.replace('_', ' ').capitalize()
            result1.append(result)
        # print(result)
        # Serialize the result, you can add additional fields
        return jsonify(result=result1, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
