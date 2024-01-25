import numpy as np
import os

os.system("cls")


class PCA:
    def __init__(self, n_component):
        self.n_component = n_component
        self.component = None
        self.mean = None
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        cov_matrix = np.cov(X.T)
        eign_vector, eign_values = np.linalg.eig(cov_matrix)
        eign_vector = eign_vector.T
        indx = np.argsort(eign_values)[::-1]
        eign_values = eign_values[indx]
        eign_vector = eign_vector[indx]
        self.component = eign_vector[:self.n_component]
        
        
    def transform(self, X):
        X = X - self.mean
        return (X.dot(self.component.T))
    
    def fit_transform(self,X):
        self.fit(X)
        self.transform(X)
