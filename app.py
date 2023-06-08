from flask import Flask, render_template, request
from PIL import Image
from segmentation.segmentator import Segmentator
from classification.classificator import Classificator
import os
from segmentation.unet import MainConv, Encoder, Decoder, UNet

app = Flask(__name__)
segmentator = Segmentator(os.getcwd())
classificator = Classificator('models/fire_classification_model.h5')


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    path = 'static'
    img_name = 'image.png'
    file.save(f'{path}/{img_name}')

    if classificator.classify(f'{path}/{img_name}') == 'Fire':
        image_path = f'{path}/{img_name}'
        img = Image.open(image_path)
        segmentator.segment(img)
        return render_template('fire_results.html'), 200

    return render_template('no_fire_results.html'), 200


if __name__ == '__main__':
    app.run()
