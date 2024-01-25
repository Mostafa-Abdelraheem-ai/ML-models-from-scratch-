import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_model(x, b):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    y_hat = x.dot(b)
    return y_hat


def find_betas(x, y):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    betas = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    return betas


def predict(x, betas):
    n = len(x)
    x_bias = np.ones((n, 1))
    x = np.c_[x_bias, x]
    prediction = x.dot(betas)
    return prediction


def square_error(y, y_hat):
    error = y - y_hat
    square = error.T.dot(error)
    return square


path = "G:\\course ML arabic Ahmed Rady\\Day 9\\real_data.csv"
data = pd.read_csv(path)

x = data[['size', 'year']]
y = data[['price']]

betas = find_betas(x, y)
y_hat = fit_model(x, betas)
sse = square_error(y, y_hat)
sst = square_error(y, y.mean())
ssr = sst - sse
print(sse)
r_square = ssr/sst
print(r_square)
