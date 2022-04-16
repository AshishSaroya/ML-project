from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
""" HELPER FUNCTION: GET ERROR RATE ========================================="""


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""


def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f' % err)


""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    print(accuracy_score(Y_test,pred_test))
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" ADABOOST IMPLEMENTATION ================================================="""


def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf_tree):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    j = 0
    for i in range(M):
        # Fit a classifier with the specific weights
        clf = clf_tree[j%4]
        j+=1
        clf.fit(X_train, Y_train, sample_weight=w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)
        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        miss2 = [x if x == 1 else 0 for x in miss]
        # Error
        err_m = np.dot(w, miss) / sum(w)
        # Alpha
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # New weights
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train,
                                          [x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,
                                         [x * alpha_m for x in pred_test_i])]
    print(len(pred_test))
    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    print(accuracy_score(Y_test,pred_test))
    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


""" PLOT FUNCTION ==========================================================="""


def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')


""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':

    # Read data
    data = pd.read_csv('cardio_train.csv', sep=';')
    data.drop('id',axis=1,inplace=True)
    data.drop_duplicates(inplace=True)
    data["bmi"] = data["weight"] / (data["height"]/100)**2
    out_filter = ((data["ap_hi"]>250) | (data["ap_lo"]>200))
    data = data[~out_filter]
    out_filter2 = ((data["ap_hi"] < 0) | (data["ap_lo"] < 0))
    data = data[~out_filter2]
    target = 'cardio'
    data_target = data[target]
    data.drop(target, axis=1, inplace=True)

    X_train, X_test, Y_train, Y_test = train_test_split(data, data_target, test_size=0.2, random_state=0)

    from sklearn.preprocessing import StandardScaler

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Fit a simple decision tree first
    clf_tree = [LogisticRegression(),GaussianNB(),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=10)]
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree[0])

    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    x_range = [3,4]
    for i in x_range:
        er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree)
        er_train.append(er_i[0])
        er_test.append(er_i[1])

    # Compare error rate vs number of iterations
    print(er_train,er_test)
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    clf.fit(X_train,Y_train)
    pred = clf.predict(X_test)
    print(accuracy_score(Y_test,pred))