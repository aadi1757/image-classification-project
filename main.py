import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from werkzeug.utils import secure_filename


app = Flask(__name__)

MODEL_PATH = 'artifacts/my_cnn_model.tf'
model = load_model(MODEL_PATH)

TARGET_SIZE = (32, 32)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)


        prediction = predict_image(file_path)

        return render_template('result.html', prediction=prediction)

    return redirect(request.url)

def predict_image(file_path):
    img = image.load_img(file_path, target_size=TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)


    predictions = model.predict(img_array)


    predicted_class = np.argmax(predictions)


    classes = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']
    predicted_class_name = classes[predicted_class]

    return predicted_class_name

if __name__ == '__main__':
    app.run(debug=True)
