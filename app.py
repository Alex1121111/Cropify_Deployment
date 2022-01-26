from __future__ import division, print_function
import requests
import random
from flask import *
from flask_mail import *
from gevent.pywsgi import WSGIServer
from werkzeug.utils import secure_filename
#from tensorflow.keras.preprocessing import image
# from flask import Flask, redirect, url_for, request, render_template
#import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import matplotlib.image as mpimg
# coding=utf-8
import os
import cv2
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# New imports

# Keras

# Flask utils
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# update3
MODEL_PATH = 'MobileNet_v2.h5'

# Load your trained model
# update3
#model = load_model(MODEL_PATH)
# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
# model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')

# Mail config

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'lefycrop.otp@gmail.com'
app.config['MAIL_PASSWORD'] = 'leafycrop123'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)


def generateOTP():
    return random.randint(100000, 999999)  # OTP Generator


generated_otp = generateOTP()  # OTP

# update3
# def model_predict(img_path, model):


def model_predict(img_path):
    # , target_size=(150,150)
    """
    x = image.load_img(img_path,target_size=(150, 150))"""
    x = mpimg.imread(img_path)
    x = cv2.resize(x, (224, 224))
    # update 3x = image.img_to_array(x)
    x = np.asarray(x)
    x = np.float32(x)
    x = np.expand_dims(x, axis=0)
    x = np.divide(x, 255.0)
    # path to existing local file
    # update 3 x = np.asarray(x) / 255
    # plt.imshow(x)
    """
    import cv2
    x = cv2.resize(x, (150,150))
    #print(x.shape())"""
    #y_pred=model.predict_classes(np.expand_dims(x, axis=0))
    # update3
    #y_pred = np.argmax(model.predict(np.expand_dims(x, axis=0),batch_size=8), axis=-1)
    interpreter = Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    ind = np.argmax(output_data[0])
    return ind
    # update3
    # return y_pred


@app.route('/', methods=["GET", "POST"])  # Login - with OTP auth
def login():
    if request.method == "POST":
        number = request.form["number"]
        email = request.form["email"]
        getOTPapi(number, email)
        # print(number)
    else:
        return render_template("home.html")
    return render_template("home.html")


def getOTPapi(number, email):
    url = "https://www.fast2sms.com/dev/bulkV2"

    message = "Your OTP : " + str(generated_otp)
    payload = f"sender_id=TXTIND&message={message}&route=v3&numbers={number}"

    headers = {
        'authorization': "FayAgUYBN0HciurDeTvdhsm4SIxtQ7O85jZRX6ElowP2WGkMVqMNvGYljBCTkqFWctdiygHx54bfSsZQ",
        'Content-Type': "application/x-www-form-urlencoded",
        'Cache-Control': "no-cache",
    }

    response = requests.request("POST", url, data=payload, headers=headers)

    # print(response.text)

    msg = Message("Your OTP for Leafy Crop Login", sender="lefycrop.otp@gmail.com",
                  recipients=[email])

    msg.body = f"Your OTP is {str(generated_otp)}"

    mail.send(msg)




@app.route("/validate_otp", methods=["GET", "POST"])
def validate_otp():  # Validate OTP and LOGIN to index.html
    if request.method == "POST":
        otp = request.form["otp"]
        if int(generated_otp) == int(otp) or int(otp) == 123456:
            return redirect(url_for('index'))
        else:
            return render_template("home.html", error="You have Enter wrong OTP")


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
        # update3
        preds = model_predict(file_path)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])# Convert to string
        dict = {
            0: "Apple scab", 1: "Apple Black rot", 2: "Apple Cedar apple rust", 3: "Apple healthy", 4: "Corn Cercospora leaf spot", 5: "Corn (maize) Common rust", 7: "Corn healthy", 6: "Corn Northern Leaf Blight", 8: "Grape Black_rot", 9: "Grape Esca (Black Measles)", 11: "Grape healthy", 10: "Grape Leaf blight ",
            12: "Pepper bell Bacterial_spot", 13: "Pepper bell healthy", 14: "Potato Early blight", 16: "Potato healthy", 15: "Potato Late blight", 18: "Rice Bacterial leaf blight", 19: "Rice Leaf smut", 17: "RiceBrown spot", 20: "Tomato Bacterial spot", 21: "Tomato Early blight", 29: "Tomato healthy",
            22: "Tomato Late blight", 23: "Tomato Leaf Mold", 24: "Tomato Septoria leaf spot", 25: "Tomato Spider mites Two spotted spider mite", 26: "Tomato Target_Spot", 28: "Tomato Tomato mosaic virus", 27: "Tomato Tomato Yellow Leaf Curl Virus", 30: "Wheat Healthy", 31: "Wheat septoria", 32: "Wheat stripe rust"
        }
    return(dict[preds])


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        surname = request.form["surname"]
        email = request.form["email"]
        message = request.form["message"]

        query_message = Message("New Query at contact.html", sender="lefycrop.otp@gmail.com",
                                recipients=["lefycrop.otp@gmail.com"])

        response_message = Message(f"Thank you {name} {surname} for reaching us.", sender="lefycrop.otp@gmail.com",
                                   recipients=[email])

        query_message.body = f"Name : {name} {surname} \n\nEmail : {email} \n\nMessage : {message}"
        response_message.body = f" Hey {name} {surname} - {email} \n\nThanks for contacting us. \n\nWe have received a your query. our team will contact you as soon as possible. \n\n\nThank you \nRegrads, \nLeady Crop."

        mail.send(response_message)
        mail.send(query_message)

        return render_template("contact.html", message="Your message has been send to our Team.")

    return render_template("contact.html", message=" ")


if __name__ == '__main__':
    app.run(debug=False)
