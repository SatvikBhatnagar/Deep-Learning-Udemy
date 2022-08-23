# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:36:10 2022

@author: satvik
"""

'''DATA PRE-PROCESSING TOOLS'''

#-------Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#-------Importing the dataset
dataset = pd.read_csv("Materials/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv")
X = dataset.iloc[:, :-1].values #iloc -> locate indexes --- iloc[rows, columns]
y = dataset.iloc[:, -1].values
#print(X)
#print(y)


#-------Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


"""Encoding categorical data"""


#-------Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


#-------Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#-------Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)


#-------Feature Scaling