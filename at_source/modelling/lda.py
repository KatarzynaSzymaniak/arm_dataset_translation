import numpy as np
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def z_score(x_train, x_test):
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test


def lda_acc(x_test, y_test, x_train, y_train, positions=None):
    x_train, x_test = z_score(x_train, x_test)

    # Create an LDA classifier
    lda = LDA()
    lda.fit(x_train, y_train.ravel())
    y_pred = lda.predict(x_test)
    accuracy = accuracy_score(y_test.ravel(), y_pred)
    #print(f"Grasp: {positions}, Accuracy:, {accuracy}")
    return accuracy