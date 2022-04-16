import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from os import path

data = pd.read_csv('cardio_train.csv',sep=';')
data.drop('id',axis=1,inplace=True)
# data.drop_duplicates(inplace=True)
# out_filter = ((data["ap_hi"]>250) | (data["ap_lo"]>200))
# data = data[~out_filter]
# data["bmi"] = data["weight"] / (data["height"]/100)**2
# data.drop('weight',axis=1,inplace=True)
# data.drop('height',axis=1,inplace=True)
# out_filter2 = ((data["ap_hi"] < 0) | (data["ap_lo"] < 0))
# data = data[~out_filter2]
target = 'cardio'
data_target = data[target]
data.drop(target,axis=1,inplace=True)

train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

# from sklearn.preprocessing import StandardScaler
# scalar = StandardScaler()
# train=scalar.fit_transform(train)
# test = scalar.transform(test)

fileName = 'Naive bayes.sav'
if(not path.exists(fileName)):
    naiveBayesClassifier = GaussianNB()
    naiveBayesClassifier.fit(train, target)
    pickle.dump(naiveBayesClassifier,open(fileName,'wb'))
naiveBayesClassifier = pickle.load(open(fileName,'rb'))

y_pred_test_naiveBayes = naiveBayesClassifier.predict(test)
print("Naive Bayes Accuracy:"+str(accuracy_score(target_test,y_pred_test_naiveBayes)*100))
cm = confusion_matrix(target_test,y_pred_test_naiveBayes)

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"]))


fileName = 'Perceptron.sav'
if(not path.exists(fileName)):
    perceptronClassifier = MLPClassifier()
    perceptronClassifier.fit(train, target)
    pickle.dump(perceptronClassifier,open(fileName,'wb'))
perceptronClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_perceptron = perceptronClassifier.predict(test)
print()
print("Perceptron accuracy:" + str(accuracy_score(target_test,y_pred_test_perceptron)*100))
cm = confusion_matrix(target_test,y_pred_test_perceptron)

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"]))

fileName = 'SVC.sav'
if(not path.exists(fileName)):
    svcClassifier = SVC()
    svcClassifier.fit(train, target)
    pickle.dump(svcClassifier,open(fileName,'wb'))
svcClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_svc = svcClassifier.predict(test)
print()
print("SVC Accuracy:"+str(accuracy_score(target_test,y_pred_test_svc)*100))
cm = confusion_matrix(target_test,y_pred_test_svc)

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"]))

fileName = 'Logistic Regression.sav'
if(not path.exists(fileName)):
    logClassifier = LogisticRegression()
    logClassifier.fit(train, target)
    pickle.dump(logClassifier,open(fileName,'wb'))
logClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_log = logClassifier.predict(test)
print()
print("Logistic Regression Accuracy:"+str(accuracy_score(target_test,y_pred_test_log)*100))
cm = confusion_matrix(target_test,y_pred_test_log)

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"]))
