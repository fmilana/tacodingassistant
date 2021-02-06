from flask import Flask, request, render_template, jsonify
from train import train
from classify import Classifier


app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/code/', methods=['POST'])
def code():
    text = request.form['text']
    train(text)
    classifier = Classifier()
    classifier.classify(text)
    output_dict = classifier.get_output_dict()
    return jsonify(output_dict)
