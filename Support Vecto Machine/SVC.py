import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.system("cls")

class HardMarginSVC:
    def __init__(self, learning_rate=.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None

    def gradient_asccent(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        for _ in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * (X.dot(X.T))
            gradinet = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradinet

        self.alpha = np.where(self.alpha < 0, 0, self.alpha)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        self.gradient_asccent(X, y)
        indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
        for i in indexes_SV:
            self.w += self.alpha[i] * y[i] * X[i]
        for i in indexes_SV:
            self.b += y[i] - np.dot(self.w.T, X[i])

        self.b /= len(indexes_SV)
    
    def predict(self, X):
        hyper_plane = X.dot(self.w) + self.b
        result = np.where(hyper_plane >=0, 1, -1)
        return result
    
    def descion_function(self,X):
        hyper_plane = X.dot(self.w) + self.b
        return hyper_plane
    
    def score(self,X,y):
        p = self.predict(X)
        return np.mean(p == y)



class SoftMarginSVC:
    def __init__(self, learning_rate=.001, iterations=1000, C=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None
        self.C = C

    def gradient_asccent(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        for _ in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * (X.dot(X.T))
            gradinet = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradinet

        self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        self.alpha = np.where(self.alpha < self.C, self.C, self.alpha)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        self.gradient_asccent(X, y)
        indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
        for i in indexes_SV:
            self.w += self.alpha[i] * y[i] * X[i]
        for i in indexes_SV:
            self.b += y[i] - np.dot(self.w.T, X[i])

        self.b /= len(indexes_SV)

    def predict(self, X):
        hyber_plane = X.dot(self.w) * self.b
        result = np.where(hyber_plane > 0, 1, -1)
        return result


class SoftMarginSVCPrimal:
    def __init__(self, learning_rate=.001, iterations=1000,C = 1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.C = C

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        for _ in range(self.iterations):
            condition = y * (X.dot(self.w)+ self.b)
            idx_miss_classifed_points = np.where(condition < 1)[0]
            d_w = self.w - self.C * y[idx_miss_classifed_points].dot(X[idx_miss_classifed_points])
            self.w -= self.learning_rate * d_w
            d_b = -self.C * np.sum(y[idx_miss_classifed_points])
            self.b -= self.learning_rate * d_b
        
    def predict(self, X):
        hyper_plane = X.dot(self.w) + self.b
        result = np.where(hyper_plane >=0, 1, -1)
        return result
    
    def descion_function(self,X):
        hyper_plane = X.dot(self.w) + self.b
        return hyper_plane

    def score(self,X,y):
        p = self.predict(X)
        return np.mean(p == y)
