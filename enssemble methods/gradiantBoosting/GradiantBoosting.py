import numpy as np
import os
os.system("cls")
from sklearn.tree import DecisionTreeRegressor

class GradiantBoostingRegressor :
    def __init__(self,n_estimators = 100,learning_rate = 0.1,max_depth=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.max_depth = max_depth
        self.y = None
        
    def fit(self,X,y):
        self.y = y
        initial_prediction = np.mean(y)
        y_hat = np.ones_like(y) * initial_prediction

        for _ in range(self.n_estimators):
            error = y - y_hat
            model = DecisionTreeRegressor(max_depth=self.max_depth)
            model.fit(X, error)
            Predicted_error = model.predict(X)
            y_hat += self.learning_rate * Predicted_error
            self.models.append(model)

            
    def predict (self,X):
        y_hat = np.mean(self.y)
        for model in self.models:
            y_hat += self.learning_rate * model.predict(X)
        return y_hat