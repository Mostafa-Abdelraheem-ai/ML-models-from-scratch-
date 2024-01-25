import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
os.system("cls")


class Softmax:
    def __init__(self, iterations=1000, alpha=.1):
        self.iterations = iterations
        self.alpha = alpha
        self.betas = None

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, keepdims=True, axis=1))
        return exp_z / np.sum(exp_z, keepdims=True, axis=1)

    def gradient_descent(self, X, y, num_classes):
        num_samples, num_features = X.shape
        self.betas = np.zeros((num_features + 1, num_classes))
        x_biased = np.c_[np.ones(num_samples), X]
        for i in range(self.iterations):
            scores = np.dot(x_biased, self.betas)
            probs = self.softmax(scores)
            gradient = np.dot(x_biased.T, (probs - y)) / num_samples
            self.betas -= self.alpha * gradient

    def fit(self, X, y):
        num_classes = np.unique(y).shape[0]
        encoder = OneHotEncoder(sparse=False)
        y_encoded = encoder.fit_transform(y.reshape(-1, 1))
        self.gradient_descent(X, y_encoded, num_classes)

    def predict(self, X):
        x_biased = np.c_[np.ones(X.shape[0]), X]
        scores = np.dot(x_biased, self.betas)
        probs = self.softmax(scores)
        predicted_labels = np.argmax(probs, axis=1)
        return predicted_labels

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
