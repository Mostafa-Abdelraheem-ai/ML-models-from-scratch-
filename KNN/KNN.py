import numpy as np
import os
os.system("cls")


class KNNclassifier:
    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.y_train = None
        

    def ecuildean_distance(self, x1, x2):
        return np.sqrt(np.sum((x2-x1)**2))

    def fit (self,x,y):
        self.x_train = x 
        self.y_train = y 
        
    def predict (self,x_new) :
        predictions = []
        for x in x_new:
            distances = [self.ecuildean_distance(x,i) for i in self.x_train ]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices] 
            common = np.bincount(k_labels).argmax()
            predictions.append(common)
        return np.array(predictions)
        
        


class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.x_train = None
        self.y_train = None
        

    def ecuildean_distance(self, x1, x2):
        return np.sqrt(np.sum((x2-x1)**2))

    def fit (self,x,y):
        self.x_train = x 
        self.y_train = y 
        
    def predict (self,x_new) :
        predictions = []
        for x in x_new:
            distances = [self.ecuildean_distance(x,i) for i in self.x_train ]
            k_indices = np.argsort(distances)[:self.k]
            k_values = [self.y_train[i] for i in k_indices] 
            result = np.mean(k_values)
            predictions.append(result)
        return np.array(predictions)