from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files['file']
    file.save('static/image.png')
    return render_template('showpicture.html'), 200


if __name__ == '__main__':
    app.run()
