'''
@Team 4: 

Manasa Kashyap Harinath
Sravanthi Avasarala
Pavitra Shivanand Hiremath
Ankit Bisht	
 
Overview: We are implementing 'Multiple linear regression' using least square method on a dataset with multiple predictors and single response variable. We are trying to predict the value of response variable based on all the predictor values.
'''

import pandas as pd
from csv import reader
import csv
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt


'''Our model is trained using train_set and test against vaidation set (This is obtained by the calculatekfoldSplit method). The model which has least RMSE is considered for the final testing. This method returns the best model along with the testing set'''
def computeLinearRegUsingKFold(X,kfl):
	betaNormkFoldArr=[]
	rmseArr=[]	
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state = 42)  
	for train, test in kfl.split(X_train):
	
		Xtest_1, Ytest_1, Xtrain_1, Ytrain_1=calculatekfoldSplit(train, test,X_train,y_train)
		betakFold=calculateTheBetaMatrix(Xtrain_1,Ytrain_1)	
		betaNormkFold=calculateBetaEachVal(betakFold)
		betaNormkFoldArr.append(betaNormkFold)
		resultkFold=calculatePreds(betaNormkFold,Xtest_1,Ytest_1)
		mkfold,nkfold = Ytest_1.shape	
		yActkFold=calculateYActForRMSE(Ytest_1,mkfold)
	        rmse=calrmse(resultkFold,yActkFold)
		rmseArr.append(rmse)
		print rmse
	
	#print 'The average RMSE using KFold is ', np.mean(rmseArr)
	#print 'hhi'
	#print betaNormkFoldArr[rmseArr.index(np.amin(rmseArr))]
	betaFinal= betaNormkFoldArr[rmseArr.index(np.amin(rmseArr))]
	#print betaFinal
	#betaNorm=calculateBetaEachVal(betaFinal)
	return betaFinal,X_test,y_test
	
'''This implementation is only for comparison purposes. Here, we using train_test_split() method for splitting the entire dataset into train
and test data and train our model using train data and test it against the test data and corresponding RMSE is calculated'''
def computeLinearRegUsingTestTrain(X,Y):

	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state = 42)  
	beta=calculateTheBetaMatrix(X_train,y_train)
	betaNorm=calculateBetaEachVal(beta)
	result=calculatePreds(betaNorm,X_test,y_test)
	m,n = y_test.shape
	yAct=calculateYActForRMSE(y_test,m)
	rmse=calrmse(result,yAct)
	print 'The root mean square error with test train split is ', rmse


def Normalize(values):
    descriptors = np.zeros((values.shape))  
    Mean = np.mean(values,axis = 1)
    SD = np.std(values,axis=1)
    size_of_descript = values.shape
    for m in range(size_of_descript[1]):

        for k in range(size_of_descript[0]):
		
            descriptors[k,m] = ((values[k,m] - Mean[k])/float(SD[k]))
		
    return descriptors

'''From the obtained weight matrix and y_test we compute result=weight_matrix * y_test'''
def calculatePreds(beta,X_test,y_test):
	
	result=np.dot(X_test,beta)	
	return result

'''Calculation of RMSE'''
def calrmse(predictions, targets):
	n = len(predictions)
	rmse = np.linalg.norm(predictions - targets) / np.sqrt(n)
	return rmse

'''This is our main model. This constitutes to Linear regression using least square method, weighted_matrix=inverseof(X'* X) * (X' * Y)'''
def calculateTheBetaMatrix(X_train,y_train):

	beta=np.dot(np.linalg.pinv(np.dot(np.transpose(X_train),X_train)),np.dot(np.transpose(X_train),y_train))
	return beta

'''Converting the matrix into weight array for easy computation'''
def calculateBetaEachVal(beta):
	betaNorm=[]	
	for eachVal in beta:
	 	betaNorm.append(eachVal[0])
	return betaNorm

'''Converting the datafrom back to array'''
def calculateYActForRMSE(y_test,m):
	yAct=[]
	for i in range(m):
	 	yAct.append(y_test.as_matrix()[i][0])
	return yAct


'''The train and test obtained from KFold.split() method are the indices. In this method, we retrieve the corresponding data from X_train 
and y_train matrices. Thus, obtained Xtest_1, Ytest_1, Xtrain_1, Ytrain_1 are returned for the weight matrix computation'''
def calculatekfoldSplit(train, test, X_train,y_train):
	
	print train.shape, test.shape	
	Xtest_1 = []
	Ytest_1 = []
	Xtrain_1=[]
	Ytrain_1 = []
	for i in range(len(test)):
		Xtest_1.append( X_train.as_matrix()[test[i]])
		Ytest_1.append( y_train.as_matrix()[test[i]])
	Xtest_1= pd.DataFrame(Xtest_1)
	Ytest_1= pd.DataFrame(Ytest_1)
	
	for i in range(len(train)):
		Xtrain_1.append( X_train.as_matrix()[train[i]])
		Ytrain_1.append( y_train.as_matrix()[train[i]])
	Xtrain_1= pd.DataFrame(Xtrain_1)
	Ytrain_1= pd.DataFrame(Ytrain_1)
	
	
	return Xtest_1, Ytest_1, Xtrain_1, Ytrain_1


#Starts from here

''' 
We start by reading the CSV file. To make it easy to understand, we have split the dataset set into two CSV files:

1) predictors.csv - contains all the predictor varibles (each column represents single predictor variable). Currently, we are considering 7 
		    predictor varibles
2) response.csv - contains only response variables.
'''

'''Using pandas library we read the csv file using read_csv method. We convert the data we obtain from read_csv to matrix form.'''
XMat = pd.read_csv('predictors.csv')
XmatrixInti = XMat.as_matrix()

#descript = Normalize(XmatrixInti)

'''We then convert thus obtained data matrix into Dataframes using pandas library '''
X = pd.DataFrame(XmatrixInti)
Y= pd.read_csv('response.csv')


''' Initializing the KFold. In our implementation, we have 5folds. '''
kfl = KFold(n_splits=5, random_state=42, shuffle=True)


'''Using kfold split, we implement our model. Please refer the computeLinearRegUsingKFold method for the detailed explanation
This method returns our model(betaNorm: Weight matrix) with x_test and y_test to test our model.'''
betaNorm,X_test,y_test=computeLinearRegUsingKFold(X,kfl)

'''Using calculatePreds method, we obtain the result matrix. This result matrix has the predicted response values'''
result=calculatePreds(betaNorm,X_test,y_test)
m,n = y_test.shape


'''Converting the y_test datafrome into Array for RMSE computation'''
yAct=calculateYActForRMSE(y_test,m)

'''Thus, RMSE is calculated and the value is recorded'''
rmse=calrmse(result,yAct)
print 'The root mean square error with KFOLD cross_validation method ', rmse

'''We used our implementation using test_train_split() instead of KFOLD, just for the comparision purpose'''
computeLinearRegUsingTestTrain(X,Y)



















