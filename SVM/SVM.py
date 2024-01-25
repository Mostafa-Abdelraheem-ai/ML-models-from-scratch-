import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.system("cls")


class HardMarginSVC:
    def __init__(self, learning_rate=0.001, iterations=1000):
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
        result = np.where(hyper_plane >= 0, 1, -1)
        return result

    def decision_function(self, X):
        hyber_plane = X.dot(self.w) + self.b
        return hyber_plane


class SoftmarginSVC:
    def __init__(self, learning_rate=.01, iterations=1000, C=1):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None
        self.C = C

    def gradient_asccent(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        for i in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * (X.dot(X.T))
            gradient = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradient
        #self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        #self.alpha = np.where(self.alpha > self.C, self.C, self.alpha)
        self.alpha = np.clip(self.alpha, 0, self.C)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        self.gradient_asccent(X, y)
        indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
        for i in indexes_SV:
            self.w += self.alpha[i] * y[i] * X[i]  # perfect values for w
        for i in indexes_SV:
            self.b += y[i] - np.dot(self.w.T, X[i])

        self.b /= len(indexes_SV)

    def predict(self, X):
        hyber_plane = X.dot(self.w) + self.b
        result = np.where(hyber_plane >= 0, 1, -1)
        return result

    def decision_function(self, X):
        hyber_plane = X.dot(self.w) + self.b
        return hyber_plane

    def score(self, X, y):
        p = self.predict(X)
        return np.mean(p == y)


class SoftmarginSVCPrimal:
    def __init__(self, learning_rate=.001, iterations=1000, C=1):
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
            condition = y * (X.dot(self.w)+self.b)
            idx_miss_classifed_points = np.where(condition < 1)[0]
            d_w = self.w - self.C * \
                (y[idx_miss_classifed_points].dot(X[idx_miss_classifed_points]))
            self.w -= self.learning_rate * d_w
            d_b = -self.C * np.sum(y[idx_miss_classifed_points])
            self.b -= self.learning_rate * d_b

    def predict(self, X):
        hyber_plane = X.dot(self.w) + self.b
        result = np.where(hyber_plane >= 0, 1, -1)
        return result

    def decision_function(self, X):
        hyber_plane = X.dot(self.w) + self.b
        return hyber_plane

    def score(self, X, y):
        p = self.predict(X)
        return np.mean(p == y)


class LinearSVM:
    def __init__(self, learning_rate=.01, iterations=1000, C=1, dual=False):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None
        self.C = C
        self.dual = dual

    def gradient_asccent(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        for i in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * (X.dot(X.T))
            gradient = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradient
        #self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        #self.alpha = np.where(self.alpha > self.C, self.C, self.alpha)
        self.alpha = np.clip(self.alpha, 0, self.C)

    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        if self.dual:
            self.gradient_asccent(X, y)
            indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
            for i in indexes_SV:
                self.w += self.alpha[i] * y[i] * X[i]  # perfect values for w
            for i in indexes_SV:
                self.b += y[i] - np.dot(self.w.T, X[i])

            self.b /= len(indexes_SV)
        else:
            for _ in range(self.iterations):
                condition = y * (X.dot(self.w)+self.b)
                idx_miss_classifed_points = np.where(condition < 1)[0]
                d_w = self.w - self.C * \
                    (y[idx_miss_classifed_points].dot(
                        X[idx_miss_classifed_points]))
                self.w -= self.learning_rate * d_w
                d_b = -self.C * np.sum(y[idx_miss_classifed_points])
                self.b -= self.learning_rate * d_b

    def predict(self, X):
        hyber_plane = X.dot(self.w) + self.b
        result = np.where(hyber_plane >= 0, 1, -1)
        return result

    def decision_function(self, X):
        hyber_plane = X.dot(self.w) + self.b
        return hyber_plane

    def score(self, X, y):
        p = self.predict(X)
        return np.mean(p == y)


class PolySVM:
    def __init__(self, learning_rate=.001,
                 iterations=1000, C=1, dual=False,
                 kernal="linear", degree=2,
                 gamma=1.0
                 ):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.alpha = None
        self.C = C
        self.dual = dual
        self.kernal = kernal
        self.degree = degree
        self.X_train = None
        self.y_train = None
        self.gamma = gamma

    def kernal_fun(self, x1, x2):
        if self.kernal == "linear":
            return x1.dot(x2.T)
        elif self.kernal == "poly":
            return (x1.dot(x2.T)+1)**self.degree
        elif self.kernal == "rbf":
            norm = np.linalg.norm(x1[:,np.newaxis]-x2, axis=2)
            return np.exp(-self.gamma * norm**2)
        else:
            raise ValueError("enter a vaild value")

    def gradient_asccent(self, X, y):
        num_samples, num_features = X.shape
        self.alpha = np.zeros(num_samples)
        kernal_matrix = self.kernal_fun(X, X)
        for i in range(self.iterations):
            y = y.reshape(-1, 1)
            H = y.dot(y.T) * kernal_matrix
            gradient = np.ones(num_samples) - H.dot(self.alpha)
            self.alpha += self.learning_rate * gradient
        #self.alpha = np.where(self.alpha < 0, 0, self.alpha)
        #self.alpha = np.where(self.alpha > self.C, self.C, self.alpha)
        self.alpha = np.clip(self.alpha, 0, self.C)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        num_samples, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        if self.kernal != 'linear':
            self.dual = True
        if self.dual:
            self.gradient_asccent(X, y)
            indexes_SV = [i for i in range(num_samples) if self.alpha[i] != 0]
            if self.kernal == "linear":
                for i in indexes_SV:
                    self.w += self.alpha[i] * y[i] * X[i]
                for i in indexes_SV:
                    self.b += y[i] - np.dot(self.w.T, X[i])
                self.b /= len(indexes_SV)
            elif self.kernal == "poly":
                for i in indexes_SV:
                    kernal_matrix = self.kernal_fun(X, X[i])
                    self.b += y[i] - np.sum(self.alpha * y * kernal_matrix)
                self.b /= len(indexes_SV)
            elif self.kernal == "rbf":
                kernal_matrix = self.kernal_fun(X, X)
                self.b = np.mean(y[indexes_SV] - np.sum(self.alpha * y *kernal_matrix[indexes_SV, :], axis=1))

        else:
            for _ in range(self.iterations):
                condition = y * (X.dot(self.w) + self.b)
                idx_miss_classifed_points = np.where(condition < 1)[0]
                d_w = self.w - self.C * \
                    y[idx_miss_classifed_points].dot(
                        X[idx_miss_classifed_points])
                self.w -= self.learning_rate * d_w
                d_b = -self.C * np.sum(y[idx_miss_classifed_points])
                self.b -= self.learning_rate * d_b

    def predict(self, X_new):
        hyber_plane = self.decision_function(X_new)
        result = np.where(hyber_plane >= 0, 1, -1)
        return result

    def decision_function(self, X_new):
        kernal_matrix = self.kernal_fun(X_new, self.X_train)
        hyber_plane = kernal_matrix.dot(self.alpha * self.y_train) + self.b
        return hyber_plane

    def score(self, X, y):
        p = self.predict(X)
        return np.mean(p == y)
