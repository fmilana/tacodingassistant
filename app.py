import json
from flask import Flask, request, render_template, jsonify
from classify import Classifier


app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/code/', methods=['POST'])
def code():
    text = request.form['text']
    classifier = Classifier()
    classifier.classify(text)
    output_dict = classifier.get_output_dict()
    dump_classification(output_dict)
    return jsonify(output_dict)


def dump_classification(dict):
    with open('classification.txt', 'w') as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)
