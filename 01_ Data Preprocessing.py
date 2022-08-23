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


#-------Encoding categorical data


#-------Encoding the Independent Variable


#-------Encoding the Dependent Variable


#-------Splitting the dataset into the Training set and Test set


#-------Feature Scaling