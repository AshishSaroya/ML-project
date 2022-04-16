from flask import Flask, render_template, flash, request, redirect, url_for,session
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField, DateField, RadioField
import numpy as np
import pandas as pd
import pickle
import datetime
from os import path
import matplotlib.pyplot as plt
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f21567d441f2b6176a'


def predictor(date_entry,gender,systolic,glucose,diastolic,cholestrol,smoke,alcohol,exercise,height,weight):

    columns = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alcohol', 'active', 'bmi']
    user_data = []
    day, month, year = map(int, date_entry.split('/'))
    birth_date = datetime.date(year, month, day)
    curr_date = datetime.date.today()
    user_data.append((curr_date - birth_date).days)

    user_data.append(int(gender))
    user_data.append(int(systolic))
    user_data.append(int(diastolic))
    user_data.append(int(cholestrol))
    user_data.append(int(glucose))
    user_data.append(int(smoke))
    user_data.append(int(alcohol))
    user_data.append(int(exercise))
    height = int(height)
    weight = int(weight)
    user_data.append(weight / (height / 100) ** 2)
    individual_predictions = pd.DataFrame()
    data = pd.DataFrame([user_data], columns=columns)
    fileName = 'Logistic Regression Stacking.sav'
    logClassifier = pickle.load(open(fileName, 'rb'))
    individual_predictions['Logistic'] = logClassifier.predict(data)

    fileName = 'Naive bayes Stacking.sav'
    nb = pickle.load(open(fileName, 'rb'))
    individual_predictions['Naive Bayes'] = nb.predict(data)

    fileName = 'Perceptron Stacking.sav'
    perc = pickle.load(open(fileName, 'rb'))
    individual_predictions['Perceptron'] = perc.predict(data)

    fileName = 'SVC Stacking.sav'
    svc = pickle.load(open(fileName, 'rb'))
    individual_predictions['SVC'] = svc.predict(data)

    fileName = 'Stacking.sav'
    stacking = pickle.load(open(fileName, 'rb'))
    output = pd.DataFrame()
    output['answer'] = stacking.predict(individual_predictions)
    return int(output.at[0,'answer'])


class ReusableForm(Form):

    @app.route("/", methods=['GET', 'POST'])
    def hello():
        form = ReusableForm(request.form)
        if session.get('result') is None:
            session['result']=-1
        return render_template('index.html',results=int(session['result']), form=form)

    @app.route("/check",methods=['POST'])
    def check():
        gender = request.form['gender']
        dateofBirth = request.form['birthday']
        systolic = request.form['sysbp']
        diastolic = request.form['diabp']
        cholestrol  = request.form['cholestrol']
        smoke = request.form['smoker']
        alcohol = request.form['alcohol']
        glucose = request.form['glucose']
        exercise = request.form['phe']
        height = request.form['height']
        weight = request.form['weight']
        session['result'] = predictor(dateofBirth,gender,systolic,glucose,diastolic,cholestrol,smoke,alcohol,exercise,height,weight)
        return redirect('/')
    
    @app.route("/clear", methods=['POST'])
    def clear():
        session.clear()
        return redirect('/')

if __name__ == "__main__":
    app.run()

