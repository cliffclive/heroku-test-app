import pickle
import numpy as np
from flask import Flask, render_template, request

# create instance of Flask class
site = Flask(__name__)

# load content for our website
html_doc = open("index.html").read()

with open("predictor/lr.pkl", "rb") as f:
    lr_model = pickle.load(f)

# create router functions for our pages
@site.route('/') # the site to route to, index/main in this case
def load_page():
    return html_doc

@site.route("/predict", methods=["POST", "GET"])
def load_predict_page():

    x_input = []
    for i in range(len(lr_model.feature_names)):
        f_value = float(
            request.args.get(lr_model.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs = lr_model.predict_proba([x_input]).flat

    return render_template('predictor.html',
                           feature_names=lr_model.feature_names,
                           x_input=x_input,
                           prediction=np.argsort(pred_probs)[::-1]
                           )
if __name__ == '__main__':
    site.run()
