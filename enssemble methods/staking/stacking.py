import os

os.system("cls")
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


"for Regrission and Classification"


class Stacking:
    def __init__(self, base_models, meta_models):
        self.base_models = base_models
        self.meta_models = meta_models

    def fit(self, X, y):
        base_prediction = []
        x_train, x_validation, y_train, y_validation = train_test_split(
            X, y, train_size=0.4, random_state=42
        )
        for model in self.base_models:
            model.fit(X=x_train, y=y_train)
            base_prediction.append(model.predict(x_validation))

        base_prediction = np.column_stack(base_prediction)
        """
        base_prediction = np.array(base_prediction)
        base_prediction = base_prediction.reshape(1,-1)
        """
        self.meta_models.fit(base_prediction, y_validation)
        """    
        def fit(self,X,y):
        X_train, X_vald, y_train, y_vald = train_test_split(X,y, test_size=.4, random_state=42)
        base_prediction = []
        for model in self.base_models:
            model.fit(X_train,y_train)
            base_prediction.append(model.predict(X_vald))

        base_prediction = np.column_stack(base_prediction)
        self.meta.fit(base_prediction, y_vald)"""

    def predict(self, X):
        base_prediction = np.column_stack(
            [model.predict(X) for model in self.base_models]
        )
        return self.meta_models.predict(base_prediction)
