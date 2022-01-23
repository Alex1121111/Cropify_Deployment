from __future__ import division, print_function
import requests
import random
from flask import *
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
# from flask import Flask, redirect, url_for, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow import keras
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# coding=utf-8
import sys
import os
import glob
import re
import cv2
import numpy as np
import pickle
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# New imports

# Keras

# Flask utils
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'MobileNet_v2.h5'

# Load your trained model
model = load_model(MODEL_PATH)
# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def generateOTP():
    return random.randint(100000, 999999)  # OTP Generator


generated_otp = generateOTP()  # OTP


def model_predict(img_path, model):
    # , target_size=(150,150)
    """
    x = image.load_img(img_path,target_size=(150, 150))"""
    x = mpimg.imread(img_path)
    x = cv2.resize(x, (224, 224))
    # path to existing local file
    x = np.asarray(x) / 255
    # plt.imshow(x)
    """
    import cv2
    x = cv2.resize(x, (150,150))
    #print(x.shape())"""
    #y_pred=model.predict_classes(np.expand_dims(x, axis=0))
    y_pred = np.argmax(model.predict(np.expand_dims(x, axis=0),batch_size=16), axis=-1)
    return y_pred


@app.route('/', methods=["GET", "POST"])  # Login - with OTP auth
def login():
    error = None
    if request.method == "POST":
        number = request.form["number"]
        getOTPapi(number)
        print(number)
    else:
        return render_template("home.html", error=error)
    return render_template("home.html")


def getOTPapi(number):
    url = "https://www.fast2sms.com/dev/bulkV2"

    message = "Your OTP : " + str(generated_otp)
    payload = f"sender_id=TXTIND&message={message}&route=v3&numbers={number}"

    headers = {
        'authorization': "FayAgUYBN0HciurDeTvdhsm4SIxtQ7O85jZRX6ElowP2WGkMVqMNvGYljBCTkqFWctdiygHx54bfSsZQ",
        'Content-Type': "application/x-www-form-urlencoded",
        'Cache-Control': "no-cache",
    }

    response = requests.request("POST", url, data=payload, headers=headers)

    print(response.text)

    if response.text:
        print("Done!")
    else:
        print("error")

    print(f"getOTPapi.py : {generated_otp}")


@app.route("/validate_otp", methods=["GET", "POST"])
def validate_otp():
    if request.method == "POST":
        otp = request.form["otp"]
        print(f"Validate_OTP.py : {generated_otp} , {otp}")
        if int(generated_otp) == int(otp):
            print("Verified")
            return redirect(url_for('index'))
        else:
            return redirect(url_for('login'))


@app.route('/index')
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])# Convert to string
        dict = {
            0: "Apple scab", 1: "Apple Black rot", 2: "Apple Cedar apple rust", 3: "Apple healthy", 4: "Corn Cercospora leaf spot", 5: "Corn (maize) Common rust", 7: "Corn healthy", 6: "Corn Northern Leaf Blight", 8: "Grape Black_rot", 9: "Grape Esca (Black Measles)", 11: "Grape healthy", 10: "Grape Leaf blight ",
            12: "Pepper bell Bacterial_spot", 13: "Pepper bell healthy", 14: "Potato Early blight", 16: "Potato healthy", 15: "Potato Late blight", 18: "Rice Bacterial leaf blight", 19: "Rice Leaf smut", 17: "RiceBrown spot", 20: "Tomato Bacterial spot", 21: "Tomato Early blight", 29: "Tomato healthy",
            22: "Tomato Late blight", 23: "Tomato Leaf Mold", 24: "Tomato Septoria leaf spot", 25: "Tomato Spider mites Two spotted spider mite", 26: "Tomato Target_Spot", 28: "Tomato Tomato mosaic virus",27:"Tomato Tomato Yellow Leaf Curl Virus",30:"Wheat Healthy",31:"Wheat septoria",32:"Wheat stripe rust"
        }
    return(dict[preds[0]])
    return None


@app.route("/contact")
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
    app.run(debug=True)
