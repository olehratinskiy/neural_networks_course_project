from flask import Flask, render_template, request
from image_edition.bw_image import get_bw_image
from segmentation.unet import UNet, Encoder, Decoder, MainConv
from segmentation.segmentator import Segmentator
import os
from PIL import Image


app = Flask(__name__)
segmentator = Segmentator(os.getcwd())


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    path = 'static'
    img_name = 'image.png'

    file.save(f'{path}/{img_name}')

    # get_bw_image(path, img_name)
    image_path = f'{os.getcwd()}\\static\\{img_name}'
    img = Image.open(image_path)
    segmentator.segment(img)

    return render_template('showpicture.html'), 200


if __name__ == '__main__':
    app.run()
