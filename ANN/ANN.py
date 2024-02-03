import numpy as np
import os

os.system("cls")


class Perceptron:
    def __init__(
        self, acvtivation="sigmoid", learning_rate=0.01, epoch=1000, use_gradiant=False
    ):
        self.activation = acvtivation
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.use_gradiant = use_gradiant
        self.weight = None
        self.bais = 0

    def _unit_step(self, z):
        return np.where(z >= 0, 1, 0)

    def _relu(self, z):
        return np.maximum(z, 0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _gradient_descnt(self, X, y):
        if self.activation == "sigmoid":
            for _ in range(self.epoch):
                y_hat = self.predict(X)
                error = y - y_hat
                self.weight += self.learning_rate * self.n_sampels/2 * X.T.dot(error * y_hat * (1-y_hat))
                self.bais += self.learning_rate * self.n_sampels/2 * np.sum(error * y_hat * (1-y_hat))
        elif self.activation == "relu":
            for _ in range(self.epoch):
                y_hat = self.predict(X)
                error = y - y_hat
                self.bais += self.learning_rate * np.sum(error * (y_hat > 0))
                self.weight += self.learning_rate * X.T.dot(error * (y_hat > 0))
        else:
            raise ValueError("only sigmoid and relu are supported")

    def _perceptron_update_rule(self, X, y):
        for _ in range(self.epoch):
            y_hat = self.predict(X)
            error = y - y_hat
            self.weight += self.learning_rate * X.T.dot(error)
            self.bais += self.learning_rate * np.sum(error)

    def fit(self, X, y):
        y = y.reshape(-1, 1)
        self.n_sampels, n_feature = X.shape
        self.weight = np.random.rand(n_feature, 1)
        if self.use_gradiant:
            self._gradient_descnt(X, y)
        else:
            self._perceptron_update_rule(X, y)

    def predict(self, X):
        z = X.dot(self.weight) + self.bais
        if self.activation == "sigmoid":
            y_hat = self._sigmoid(z)
        elif self.activation == "relu":
            y_hat = self._relu(z)
        elif self.activation == "unit_step":
            y_hat = self._unit_step(z)
        else:
            raise ValueError("only sigmoid, relu and unit_step are supported")
        return y_hat
