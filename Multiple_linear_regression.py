import pandas as pd
from csv import reader
import csv
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt



def computeLinearRegUsingKFold(X,kfl):
	betaNormkFoldArr=[]
	rmseArr=[]	
	for train, test in kfl.split(X):
	
		Xtest_1, Ytest_1, Xtrain_1, Ytrain_1=calculatekfoldSplit(train, test)
		betakFold=calculateTheBetaMatrix(Xtrain_1,Ytrain_1)	
		betaNormkFold=calculateBetaEachVal(betakFold)
		#betaNormkFoldArr.append(betaNormkFold)
		resultkFold=calculatePreds(betaNormkFold,Xtest_1,Ytest_1)
		mkfold,nkfold = Ytest_1.shape	
		yActkFold=calculateYActForRMSE(Ytest_1,mkfold)
	        rmse=calrmse(resultkFold,yActkFold)
		rmseArr.append(rmse)
		print rmse
	
	print 'The average RMSE is ', np.mean(rmseArr)
	#print 'hhi'
	#betaFinal= betaNormkFoldArr[rmseArr.index(np.amin(rmseArr))]
	#betaNorm=calculateBetaEachVal(betaFinal)
	#return betaNorm
	

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

def calculatePreds(beta,X_test,y_test):
	
	result=np.dot(X_test,beta)	
	return result

def calrmse(predictions, targets):
	n = len(predictions)
	rmse = np.linalg.norm(predictions - targets) / np.sqrt(n)
	return rmse

def calculateTheBetaMatrix(X_train,y_train):

	beta=np.dot(np.linalg.pinv(np.dot(np.transpose(X_train),X_train)),np.dot(np.transpose(X_train),y_train))
	return beta


def calculateBetaEachVal(beta):
	betaNorm=[]	
	for eachVal in beta:
	 	betaNorm.append(eachVal[0])
	return betaNorm


def calculateYActForRMSE(y_test,m):
	yAct=[]
	for i in range(m):
	 	yAct.append(y_test.as_matrix()[i][0])
	return yAct



def calculatekfoldSplit(test, train):
	Xtest_1 = []
	Ytest_1 = []
	Xtrain_1=[]
	Ytrain_1 = []
	for i in range(len(test)):
		Xtest_1.append( X.as_matrix()[test[i]])
		Ytest_1.append( Y.as_matrix()[test[i]])
	Xtest_1= pd.DataFrame(Xtest_1)
	Ytest_1= pd.DataFrame(Ytest_1)
	
	for i in range(len(train)):
		Xtrain_1.append( X.as_matrix()[train[i]])
		Ytrain_1.append( Y.as_matrix()[train[i]])
	Xtrain_1= pd.DataFrame(Xtrain_1)
	Ytrain_1= pd.DataFrame(Ytrain_1)
	
	
	return Xtest_1, Ytest_1, Xtrain_1, Ytrain_1


k_fold=5
XMat = pd.read_csv('predictors.csv')
XmatrixInti = XMat.as_matrix()

descript = Normalize(XmatrixInti)
X = pd.DataFrame(descript)
Y= pd.read_csv('response.csv')

kfl =KFold(n_splits=5, random_state=42, shuffle=False)

computeLinearRegUsingKFold(X,kfl)
#result=calculatePreds(betaNorm,X_test,y_test)
#m,n = y_test.shape
#yAct=calculateYActForRMSE(y_test,m)
#rmse=calrmse(result,yAct)
#print 'The root mean square error with best value ', rmse


computeLinearRegUsingTestTrain(X,Y)



















