import numpy as np
import os
os.system("cls")


class Node:
    def __init__(self, feature = None, value = None, left= None , right= None,result = None):
        #for descision node
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right

        #for leaf node
        self.result = result


class DecisionTree:
    def __init__ (self, max_depth = 3,criterion = "entropy"):
        self.max_depth = max_depth
        self.tree = None
        self.criterion = criterion

    def split_data(self, X, feature, value):
        left_indices = np.where(X[: , feature] <= value)[0]
        right_indices = np.where(X[: , feature] > value)[0]
        return left_indices, right_indices
    
    def entropy(self, y):
        p = np.bincount(y) / len(y)
        return - np.sum(p * np.log2 (p +1e-10))
    
    def gini(self,y):
        p = np.bincount(y) / len(y)
        return 1 - np.sum(p**2)

    def info_gain(self , y, left_indices, right_indices):

        if self.criterion == "entropy":
            impurity_function = self.entropy
        elif self.criterion == "gini":
            impurity_function = self.gini

        parent_impurity = impurity_function(y)
        left_impurity = impurity_function(y[left_indices])
        right_impurity = impurity_function(y[right_indices])

        w_left = len(left_indices) / len(y)
        w_right = len(right_indices) / len(y)

        information_gain = parent_impurity - w_left * left_impurity - w_right * right_impurity
        return information_gain
    
    def fit(self, X, y, depth = 0):

        if depth == self.max_depth or np.all(y[0] == y):
            return Node(result= y[0])
        
        num_samples, num_feature = X.shape
        best_information_gain = 0.0
        best_split = None
        for i in range(num_feature):
            feature_value = X[:, i]
            unique_values = np.unique(feature_value)
            for j in unique_values:
                left_indices, right_indices = self.split_data(X , i, j)
                if len(left_indices) == 0 or len(right_indices) ==0:
                    continue 
                information_gain = self.info_gain(y, left_indices , right_indices)
                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    best_split = (i , j , left_indices, right_indices)
        
        if best_information_gain == 0.0:
            return Node(result= np.bincount(y).argmax())
        
        feature, value , left_indices, right_indices = best_split
        left_subtree = self.fit(X[left_indices], y[left_indices],depth + 1)
        right_subtree = self.fit(X[right_indices], y[right_indices], depth + 1)
        self.tree = Node(feature= feature , value= value, left= left_subtree, right= right_subtree)
        return self.tree
    
    def predict(self, X):
        result = [self.predict_recursive(x,self.tree) for x in X]
        return np.array(result)

    def predict_recursive(self, x, node):
        if node.result is not None:
            return node.result
        
        if x[node.feature] <= node.value:
            return self.predict_recursive(x, node.left)
        else:
            return self.predict_recursive(x, node.right)