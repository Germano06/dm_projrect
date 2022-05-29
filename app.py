from flask import Flask, render_template, request
import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr

app = Flask(__name__)

# IMAGE_PATH = './images/'


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(imagepath)

    text = result[0][1]
    output = ''

    for detection in result:
        text = detection[1]
        output += (text+' ')

    return render_template('index.html', prediction=output)


if __name__ == '__main__':
    app.run(debug=True)
