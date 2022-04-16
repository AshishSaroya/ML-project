import numpy as np
import pandas as pd
import pickle
import datetime
from os import path
import matplotlib.pyplot as plt

columns = ['age','gender','ap_hi','ap_lo','cholesterol','gluc','smoke','alcohol','active','bmi']
user_data = []
date_entry = input("Enter date in dd/mm/yyyy format")
day,month,year = map(int,date_entry.split('/'))
birth_date = datetime.date(year,month,day)
curr_date = datetime.date.today()
user_data.append((curr_date-birth_date).days)

user_data.append(int(input("Enter 1 for woman and 2 for man")))
user_data.append(int(input("Enter Systolic Blood pressure")))
user_data.append(int(input("Enter Diastolic Blood pressure")))
user_data.append(int(input("Enter cholesterol: 1 for normal, 2 for above normal and 3 for well above normal")))
user_data.append(int(input("Enter Glucose: 1 for normal, 2 for above normal and 3 for well above normal")))
user_data.append(int(input("Enter 1 if user smokes and 0 if he doesn't")))
user_data.append(int(input("Enter 1 if user drinks alcohol and 0 if he doesn't")))
user_data.append(int(input("Enter 1 if user does physical exercise and 0 if he doesn't")))
height = int(input("Enter user's height in cm"))
weight = int(input("Enter user's weight in kg"))
user_data.append(weight/(height/100)**2)
individual_predictions = pd.DataFrame()
data = pd.DataFrame([user_data],columns = columns)
fileName = 'Logistic Regression Stacking.sav'
logClassifier = pickle.load(open(fileName,'rb'))
individual_predictions['Logistic'] = logClassifier.predict(data)

fileName = 'Naive bayes Stacking.sav'
nb = pickle.load(open(fileName,'rb'))
individual_predictions['Naive Bayes'] = nb.predict(data)

fileName = 'Perceptron Stacking.sav'
perc = pickle.load(open(fileName,'rb'))
individual_predictions['Perceptron'] = perc.predict(data)

fileName = 'SVC Stacking.sav'
svc = pickle.load(open(fileName,'rb'))
individual_predictions['SVC'] = svc.predict(data)

print(individual_predictions)
fileName = 'Stacking.sav'
stacking = pickle.load(open(fileName,'rb'))
output = pd.DataFrame()
output['answer'] = stacking.predict(individual_predictions)
print(output)
