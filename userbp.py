from flask import *
import joblib
import numpy as np
from datetime import datetime
import pickle
import pandas as pd
from views import preprocess
import routes.predict


user_bp = Blueprint('user_bp', __name__)


@user_bp.route('/user')
def user():
    return render_template("user.html")


@user_bp.route('/user_home',  methods=['POST', 'GET'])
def user_home():
    msg = ''
    print("hi")
    if request.form['user'] == 'user' and request.form['pwd'] == 'user':
        return render_template("predict.html")
    else:
        msg = 'Incorrect username / password !'
    return render_template('user.html', msg=msg)




@user_bp.route('/predict',  methods=['GET', 'POST'])
def predict():

    pred_cat=""
    pred_desc=""
    text1 = request.form['text1'].lower()

    if request.method == 'POST':

        if preprocess() == "valid":

            pred_cat, pred_desc=routes.predict.predict_tweet(text1)
            return render_template('predict.html', prediction_category=pred_cat, prediction_description=pred_desc)
        else:

            return render_template('predict.html')
    else:

        return render_template('predict.html')





@user_bp.route('/userlogout')
def userlogout():
    return render_template("home.html")