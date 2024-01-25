import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def draw_line (x,b0,b1) :
    n=len(x)

    y_hat = np.zeros(n)
    for i in range (n) :
        y_hat[i] =  b0 + b1*x[i]
    return y_hat

def calc_squard_error (x,y,b0,b1) :
    error = 0
    for i in range (len(x)):
        error += (y[i] - (b0+b1*x[i]))**2
    return error

def comput_b1 (x,y,x_mean,y_mean) :
    nomenator = 0
    domenator = 0
    for i in range (len(x)):
        nomenator = (x[i]-x_mean)-(y[i]-y_mean)
    for i in range (len(x)):
        domenator = (x[i]-x_mean)**2
    return (nomenator/domenator)


def comput_b0 (b1,x_mean,y_mean):
    b0 = y_mean - b1*x_mean
    return b0 

def predict (x , b0 , b1) :
    prediction = b0 + x*b1
    return prediction

path = "G:\\course ML arabic Ahmed Rady\\Day 7\\real data.csv"
data = pd.read_csv(path)    

x = data["SAT"]
y = data["GPA"]

x_mean = np.mean(x)
y_mean = np.mean(y)



plt.scatter(x,y)
b1 = comput_b1(x,y,x_mean,y_mean)
b0 = comput_b0(b1 , x_mean ,y_mean)

y_hat=draw_line(x,b0,b1)
plt.plot(x,y_hat)
plt.show()

print (calc_squard_error(x,y,b0,b1))
print(predict(2000,b0,b1))