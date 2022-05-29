from crypt import methods
from flask import Flask, render_template, request
import cv2
import numpy as np
from matplotlib import pyplot as plt
import easyocr
import pytesseract
from PIL import Image

app = Flask(__name__)


def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)


# def thin_font(image):
#     import numpy as np
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2, 2), np.uint8)
#     image = cv2.erode(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return (image)


# def thick_font(image):
#     import numpy as np
#     image = cv2.bitwise_not(image)
#     kernel = np.ones((2, 2), np.uint8)
#     image = cv2.dilate(image, kernel, iterations=1)
#     image = cv2.bitwise_not(image)
#     return (image)


def remove_borders(image):
    contours, heiarchy = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    return (crop)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


# @app.route('/easyOCR', methods=['GET', 'POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     imagepath = "./images/" + imagefile.filename
#     imagefile.save(imagepath)

#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(imagepath)

#     text = result[0][1]
#     output = ''

#     for detection in result:
#         text = detection[1]
#         output += (text+' ')

#     return render_template('index.html', prediction=output)


@app.route('/', methods=['GET', 'POST'])
def tasser():
    ocr_result = ''

    imagefile = request.files['imagefile']
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)

    img = cv2.imread(imagepath)
    inverted_img = cv2.bitwise_not(img)
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh, im_bw = cv2.threshold(grey_img, 210, 230, cv2.THRESH_BINARY)

    no_noise = noise_removal(im_bw)
    no_boders = remove_borders(no_noise)

    ocr_result += pytesseract.image_to_string(no_boders)

    return render_template('index.html', prediction=ocr_result)


if __name__ == '__main__':
    app.run(debug=True)
