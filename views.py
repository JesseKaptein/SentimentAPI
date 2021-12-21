import pandas as pd

from flask import Blueprint
from flask import render_template, request
from flask import jsonify

from modeling import sentiment_model


app = Blueprint('main', __name__)


@app.route("/",  methods=["GET", "POST"])
@app.route("/index",  methods=["GET", "POST"])
def index():
    """Returns homepage with sentiment prediction interface"""
    text = request.form.get('textbox')
    base_url = request.base_url

    if text:
        r = get_sentiment_probs(text)
        show_alert = 'block' if get_show_alert(text) else 'none'
        return render_template("index.html", input=text, output=r.json, base_url=base_url, show_alert=show_alert)

    return render_template("index.html", input="", output="", base_url=base_url, show_alert='none')


@app.route('/sentiment/<text>')
def get_sentiment_probs(text):
    """Endpoint returning JSON with sentiment probability scores for provided <text>"""
    text = filter(None, text.split(';'))
    probs = sentiment_model.get_prediction_probs(pd.Series(text))
    return jsonify(probs)


@app.route('/vocabulary/<text>')
def get_proc_in_vocabulary(text):
    """Endpoint returning JSON with percentage of words in <text> in model vocabulary"""
    text = filter(None, text.split(';'))
    procs = sentiment_model.proc_in_vocabulary(pd.Series(text))
    return jsonify(procs)


def get_show_alert(text):
    """Returns boolean whether or not to show alert message if percentage of words in model vocabulary is to low"""
    r = get_proc_in_vocabulary(text)
    procs = r.json
    return sum([p < 0.5 for p in procs]) > 0
