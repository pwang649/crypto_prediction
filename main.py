# Name: Peter Wang
# Semester: Fall 2021
# Section: itp_216_32080
# Assignment: Final Project
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import san
from sklearn.svm import SVR
from flask import Flask, redirect, render_template, request, session, url_for, send_file
import os
import sqlite3 as sl

app = Flask(__name__)

@app.errorhandler(404)
def not_found(e):
    return render_template("404.html")

# GET request endpoint 1
@app.route("/")
def home():
    return render_template("index.html")

# GET request endpoint 2
@app.route("/eth", methods=["POST", "GET"])
def eth_home():
    p = ''
    if 'plot' in request.args.keys():
        p = request.args['plot']
    err = ''
    if 'error' in request.args.keys():
        err = request.args['error']
    return render_template("eth.html", plot=p, error=err)

# GET request endpoint 3
@app.route("/btc", methods=["POST", "GET"])
def btc_home():
    p = ''
    if 'plot' in request.args.keys():
        p = request.args['plot']
    err = ''
    if 'error' in request.args.keys():
        err = request.args['error']
    return render_template("btc.html", plot=p, error=err)

# POST request endpoint 1
@app.route("/eth/predict", methods=["POST", "GET"])
def predict_eth():
    # load eth data from database
    ohlc_df = read_from_db('eth')
    prediction_days = 0
    if request.method == "POST":
        # error handling mechanism that prevents letter input
        try:
            prediction_days = int(request.form["prediction_days"])
            start_year = int(request.form["start_year"])
            ohlc_df = ohlc_df[(start_year - 2017) * 365:]
        except:
            return redirect(url_for("eth_home", error="Please input an integer!"))
        # Restrict the scope of prediction days to make the prediction more accurate
    if prediction_days < 30 or prediction_days > 200:
        return redirect(url_for("eth_home", error="Prediction days must be between 30 and 200!"))
    return redirect(url_for("eth_home", plot=main_prediction(ohlc_df, prediction_days, 'ETH')))

# POST request endpoint 2
@app.route("/btc/predict", methods=["POST", "GET"])
def predict_btc():
    # load btc data from database
    ohlc_df = read_from_db('btc')
    prediction_days = 0
    if request.method == "POST":
        # error handling mechanism that prevents letter input
        try:
            prediction_days = int(request.form["prediction_days"])
            start_year = int(request.form["start_year"])
            ohlc_df = ohlc_df[(start_year-2017)*365:]
        except:
            return redirect(url_for("btc_home", error="Please input an integer!"))
    # Restrict the scope of prediction days to make the prediction more accurate
    if prediction_days < 30 or prediction_days > 200:
        return redirect(url_for("btc_home", error="Prediction days must be between 30 and 200!"))
    return redirect(url_for("btc_home", plot=main_prediction(ohlc_df, prediction_days, 'BTC')))

# make a prediction based on past crypto prices and produce a prediction of future output
def main_prediction(ohlc_df, prediction_days, crypto_type):
    ohlc_df['Prediction'] = ohlc_df[['openPriceUsd']].shift(-prediction_days)
    X = np.array(ohlc_df.drop(['Prediction'], axis=1))
    X = X[:len(ohlc_df) - prediction_days]
    y = np.array(ohlc_df['Prediction'])
    y = y[:-prediction_days]

    # Create and train the Support Vector Machine
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.00001)  # Create the model

    # Here I didn't split the data into training and testing set, since all data are valuable to us
    svr_rbf.fit(X, y)  # Train the model

    # Use the same set to predict the future prices based on our model
    svm_prediction = svr_rbf.predict(X)

    return generate_img(X, svm_prediction, prediction_days, crypto_type)

# generate a 4-plot image, based on the original and ML trained data
def generate_img(X, svm_prediction, prediction_days, crypto_type):
    # create a 2 by 2 subplot, and set data, titles, axis labels, legends, etc.
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("Crypto Trend")
    plt.setp(ax[:2, :2], xlabel='time', ylabel='price')
    ax[0, 0].set_title('Original chart')
    ax[0, 0].plot(X, label=crypto_type)
    ax[0, 0].legend(loc="upper left")
    ax[0, 1].set_title('Overall ML predicted chart')
    ax[0, 1].plot(svm_prediction, label=crypto_type)
    ax[0, 1].legend(loc="upper left")
    ax[1, 0].set_title('Zoomed in ML prediction')
    ax[1, 0].plot(svm_prediction[int(svm_prediction.size/2):], label=crypto_type)
    ax[1, 0].legend(loc="upper left")
    ax[1, 1].set_title('Future prediction from now')
    ax[1, 1].plot(svm_prediction[-prediction_days:], label=crypto_type)
    ax[1, 1].legend(loc="upper left")
    plt.tight_layout()

    # save and return the resulting plot as a png image
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    png = base64.b64encode(img.getvalue())
    fig_str = str(png, 'utf-8')
    html = '<img src=\"data:image/png;base64,{}\"/>'.format(fig_str)
    return html

# read crypto data from the database on input crypto type
def read_from_db(crypto):
    # connect and grab eth data from database
    db = "database.db"
    conn = sl.connect(db)
    ohlc_df = pd.read_sql('SELECT * FROM ' + crypto, conn)
    ohlc_df.drop(['datetime'], axis=1, inplace=True)
    return ohlc_df

# Only need to run once for the entire time
# use pandas to store the crypto pricing data and save in database
def initialize_eth_db():
    # data used are from 2017 to 2021 with the time interval of 1 day
    ohlc_df = san.get(
        "ohlc/ethereum",
        from_date="2017-01-01",
        to_date="2021-12-01",
        interval="1d"
    )
    # remove unnecessary columns
    ohlc_df.drop(['closePriceUsd', 'highPriceUsd', 'lowPriceUsd'], axis=1, inplace=True)
    # connect to database and store data
    db = "database.db"
    conn = sl.connect(db)
    ohlc_df.to_sql('eth', con=conn)

def initialize_btc_db():
    # data used are from 2017 to 2021 with the time interval of 1 day
    ohlc_df = san.get(
        "ohlc/bitcoin",
        from_date="2017-01-01",
        to_date="2021-12-01",
        interval="1d"
    )
    # remove unnecessary columns
    ohlc_df.drop(['closePriceUsd', 'highPriceUsd', 'lowPriceUsd'], axis=1, inplace=True)
    # connect to database and store data
    db = "database.db"
    conn = sl.connect(db)
    ohlc_df.to_sql('btc', con=conn)

if __name__ == "__main__":
    # uncomment and run the two lines below only when database is just created
    # initialize_eth_db()
    # initialize_btc_db()
    app.secret_key = os.urandom(12)
    app.run(debug=True)