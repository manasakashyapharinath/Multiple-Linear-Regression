import pandas as pd
import numpy as np
import csv
from csv import reader
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation 
from sklearn.cross_validation import train_test_split

XMat = pd.read_csv('predictors.csv')
XmatrixInti = XMat.as_matrix()
X = pd.DataFrame(XmatrixInti)
Y= pd.read_csv('response.csv')
Ymatr=Y.as_matrix()
linreg = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(
    X, Y,test_size=0.3, random_state=0)
linreg.fit(X_train, y_train)
p=linreg.predict(X_test)
s = cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=42)
scores = cross_validation.cross_val_score(linreg, X, Y, cv=s, scoring='mean_squared_error')
print('this is second one')
print scores
scores1= -scores
rmse_scores = np.sqrt(scores1)
print('this is root mean squares ')
print(rmse_scores)
print('this is   average root mean squares ')
print(rmse_scores.mean())



