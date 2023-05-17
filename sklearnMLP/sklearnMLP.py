#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#main
if __name__ == "__main__":
    # read training & test files, seperate by space
    train_set = pd.read_csv("wine_training", sep=" ")
    test_set = pd.read_csv("wine_test", sep=" ")
    
    # MLP trains on x(feature vectors) & y(classe labels) arrays
    x_train = train_set.drop("Class", axis=1)
    y_train = train_set["Class"]
    x_test = test_set.drop("Class", axis=1)
    y_test = test_set["Class"]
    
    # fit only on training data
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    # apply same transformation to test data
    x_test = scaler.transform(x_test)
    
    # train a model
    mlp = MLPClassifier(max_iter=100)
    mlp.fit(x_train, y_train)
    
    # mlp prediction
    prediction = mlp.predict(x_test)
    
    print(classification_report(y_test, prediction))

