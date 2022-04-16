import numpy as np
import pandas as pd
import pickle
from os import path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('cardio_train.csv',sep=';')
data.drop('id',axis=1,inplace=True)
data.drop_duplicates(inplace=True)
out_filter = ((data["ap_hi"]>250) | (data["ap_lo"]>200))
data = data[~out_filter]
data["bmi"] = data["weight"] / (data["height"]/100)**2
data.drop('weight',axis=1,inplace=True)
data.drop('height',axis=1,inplace=True)
out_filter2 = ((data["ap_hi"] < 0) | (data["ap_lo"] < 0))
data = data[~out_filter2]
target = 'cardio'
data_target = data[target]
data1 = data.copy()
data.drop(target,axis=1,inplace=True)

classifier_outputs_train = pd.DataFrame()
classifier_outputs_test = pd.DataFrame()
train, test, target, target_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
train=scalar.fit_transform(train)
test = scalar.transform(test)

fileName = 'Naive bayes Bagging.sav'
if(not path.exists(fileName)):
    naiveBayesClassifier = GaussianNB()
    X_train, cvtrain, Y_train, cvtarget = train_test_split(train, target, test_size=0.25)
    naiveBayesClassifier.fit(X_train, Y_train)
    pickle.dump(naiveBayesClassifier,open(fileName,'wb'))
naiveBayesClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_naiveBayes = naiveBayesClassifier.predict(test)
classifier_outputs_test['Naive Bayes'] = y_pred_test_naiveBayes

fileName = 'Perceptron Bagging.sav'
if(not path.exists(fileName)):
    perceptronClassifier = MLPClassifier()
    X_train, cvtrain, Y_train, cvtarget = train_test_split(train, target, test_size=0.25)
    perceptronClassifier.fit(X_train, Y_train)
    pickle.dump(perceptronClassifier, open(fileName, 'wb'))
perceptronClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_perceptron = perceptronClassifier.predict(test)
classifier_outputs_test['Perceptron'] = y_pred_test_perceptron

fileName = 'SVC Bagging.sav'
if(not path.exists(fileName)):
    svcClassifier = SVC()
    X_train, cvtrain, Y_train, cvtarget = train_test_split(train, target, test_size=0.25)
    svcClassifier.fit(X_train, Y_train)
    pickle.dump(svcClassifier,open(fileName,'wb'))
svcClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_svc = svcClassifier.predict(test)
classifier_outputs_test['SVC'] = y_pred_test_svc

fileName = 'Logistic Regression Bagging.sav'
if(not path.exists(fileName)):
    logClassifier = LogisticRegression()
    X_train, cvtrain, Y_train, cvtarget = train_test_split(train, target, test_size=0.25)
    logClassifier.fit(X_train, Y_train)
    pickle.dump(logClassifier,open(fileName,'wb'))
logClassifier = pickle.load(open(fileName,'rb'))
y_pred_test_log  = logClassifier.predict(test)
classifier_outputs_test['Logistic'] = y_pred_test_log

fileName = 'Decision Tree Bagging.sav'
if(not path.exists(fileName)):
    decisionTree = DecisionTreeClassifier()
    X_train, cvtrain, Y_train, cvtarget = train_test_split(train, target, test_size=0.25)
    decisionTree.fit(X_train, Y_train)
    pickle.dump(decisionTree,open(fileName,'wb'))
decisionTree = pickle.load(open(fileName,'rb'))
y_pred_test_tree = decisionTree.predict(test)
classifier_outputs_test['Decision Tree'] = y_pred_test_tree

test_results = []
for i in range(len(test)):
    test_results.insert(i, classifier_outputs_test.iloc[i]['Logistic'] + classifier_outputs_test.iloc[i]['Naive Bayes'] + \
                       classifier_outputs_test.iloc[i]['Perceptron'] + classifier_outputs_test.iloc[i]['SVC'] + \
                       classifier_outputs_test.iloc[i]['Decision Tree'])
    if test_results[i] >= 2:
        test_results[i] = 1
    else:
        test_results[i] = 0

test_prediction = np.array(test_results)

print("Bagging ensemble accuracy:"+str(accuracy_score(target_test,test_prediction)*100))
cm = confusion_matrix(target_test,test_prediction)

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]
Precision = TP/(TP+FP)
Recall = TP/(TP+FN)
F1_Score = 2*(Recall * Precision) / (Recall + Precision)
print(pd.DataFrame([[Precision, Recall, F1_Score]],columns=["Precision", "Recall", "F1 Score"], index=["Results"]))
