from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import os
os.system("cls")


class Softmax:
    def __init__(self, iterations=1000, alpha=.1):
        self.iterations = iterations
        self.alpha = alpha
        self.betas = None

    def softmax(self, z):
        exp_z = np.exp(z-np.max(z, keepdims=True, axis=1))
        return exp_z / np.sum(exp_z, keepdims=True, axis=1)

    def grediant_descent(self, x, y, number_classes):
        number_sampels, number_features = x.shape
        betas = np.zeros((number_features+1, number_classes))
        for i in range(self.iterations):
            x_baised = np.c_[np.ones(number_sampels), x]
            scores = np.dot(x_baised, betas)
            probs = self.softmax(scores)

            grediant = np.dot(x_baised.T, (y-probs)) / number_sampels
            betas -= self.alpha * grediant

    def fit(self, x, y):
        num_classes = np.unique(y).shape[0]
        one_hot_encoder = OneHotEncoder(sparse_output=False)
        encoded_y = one_hot_encoder.fit_transform(y.reshape(-1, 1))
        self.grediant_descent(x, encoded_y, num_classes)

    def predict(self, x):
        x_baised = np.c_[np.ones(x.shape[0], x)]
        scores = np.dot(x_baised, self.betas)
        probs = self.softmax(scores)
        predicted_label = np.argmax(probs)
        return predicted_label

    def predict_proba(self, X):
        x_biased = np.c_[np.ones(X.shape[0]), X]
        scores = np.dot(x_biased, self.betas)
        probs = self.softmax(scores)
        return probs

    def score(self, X, y):
        x_biased = np.c_[np.ones(X.shape[0]), X]
        scores = np.dot(x_biased, self.betas)
        probs = self.softmax(scores)
        predicted_labels = np.argmax(probs, axis=1)
        accuracy = np.mean(predicted_labels == y)
        return accuracy
