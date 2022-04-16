import numpy as np
import pandas as pd
import pickle
from os import path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('heart.csv')
print(data)
scaler = StandardScaler()
X = scaler.fit_transform(data.drop(['target'],axis=1))
Y = data.target



classifier_outputs_train = pd.DataFrame()
classifier_outputs_test = pd.DataFrame()
train, test, target, target_test = train_test_split(X, Y, test_size=0.2, random_state=0)
test, cvtrain, target_test,cvtarget = train_test_split(test,target_test, test_size=0.5, random_state=0)


scalar = StandardScaler()
train=scalar.fit_transform(train)
cvtrain = scalar.transform(cvtrain)
test = scalar.transform(test)

fileName = 'Naive bayes Stacking1.sav'
if(not path.exists(fileName)):
    naiveBayesClassifier = GaussianNB()
    naiveBayesClassifier.fit(train, target)
    pickle.dump(naiveBayesClassifier,open(fileName,'wb'))
naiveBayesClassifier = pickle.load(open(fileName,'rb'))
y_pred_naiveBayes = naiveBayesClassifier.predict(cvtrain)
y_pred_test_naiveBayes = naiveBayesClassifier.predict(test)
classifier_outputs_train['Naive Bayes'] = y_pred_naiveBayes
classifier_outputs_test['Naive Bayes'] = y_pred_test_naiveBayes

fileName = 'Perceptron Stacking1.sav'
if(not path.exists(fileName)):
    perceptronClassifier = MLPClassifier()
    perceptronClassifier.fit(train, target)
    pickle.dump(perceptronClassifier,open(fileName,'wb'))
perceptronClassifier = pickle.load(open(fileName,'rb'))
y_pred_perceptron = perceptronClassifier.predict(cvtrain)
y_pred_test_perceptron = perceptronClassifier.predict(test)
classifier_outputs_train['Perceptron'] = y_pred_perceptron
classifier_outputs_test['Perceptron'] = y_pred_test_perceptron


fileName = 'SVC Stacking1.sav'
if(not path.exists(fileName)):
    svcClassifier = SVC()
    svcClassifier.fit(train, target)
    pickle.dump(svcClassifier,open(fileName,'wb'))
svcClassifier = pickle.load(open(fileName,'rb'))
y_pred_svc = svcClassifier.predict(cvtrain)
y_pred_test_svc = svcClassifier.predict(test)
classifier_outputs_train['SVC'] = y_pred_svc
classifier_outputs_test['SVC'] = y_pred_test_svc

fileName = 'Logistic Regression Stacking1.sav'
if(not path.exists(fileName)):
    logClassifier = LogisticRegression()
    logClassifier.fit(train, target)
    pickle.dump(logClassifier,open(fileName,'wb'))
logClassifier = pickle.load(open(fileName,'rb'))
y_pred_log = logClassifier.predict(cvtrain)
y_pred_test_log  = logClassifier.predict(test)
classifier_outputs_train['Logistic'] = y_pred_log
classifier_outputs_test['Logistic'] = y_pred_test_log

fileName = "Stacking1.sav"
if(not path.exists(fileName)):
    classifier = DecisionTreeClassifier()
    classifier.fit(classifier_outputs_train, cvtarget)
    pickle.dump(classifier, open(fileName, 'wb'))
classifier = pickle.load(open(fileName,'rb'))
y_pred = classifier.predict(classifier_outputs_test)

print("Stacking ensembling accuracy:"+str(accuracy_score(target_test,y_pred)*100))
cm = confusion_matrix(target_test,y_pred)

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"]))