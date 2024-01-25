import numpy as np
import os
os.system("cls")

class KNNClassifierA:
    def __init__ (self, k = 3):
        self.k = k
        self.X_train = None
        self .y_train = None

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum( (x2 - x1) ** 2))


    def fit (self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_new):
        predictions = []
        for x in X_new:
            distances = [self.euclidean_distance(x,i) for i in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices] 
            common = np.bincount(k_labels).argmax()
            predictions.append(common)
        return np.array(predictions)



class KNNRegressor:
    def __init__ (self, k = 3):
        self.k = k
        self.X_train = None
        self .y_train = None

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum( (x2 - x1) ** 2))
    
    def fit (self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_new):
        predictions = []
        for x in X_new:
            distances = [self.euclidean_distance(x,i) for i in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_values = [self.y_train[i] for i in k_indices]
            result = np.mean(k_values)
            predictions.append(result)
        return np.array(predictions)

