# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 07:55:39 2019

@author: Akshay Kumar C P
"""
'''
Refered krish naik's code for learning purpose
'''

'''
import lib's and dataset
'''

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')


'''
EDA
'''
train.head()

# missing data finding

train.columns

train.isnull()

sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sb.set_style('whitegrid')
sb.countplot(x='Survived',data=train)

sb.countplot(x='Survived',hue='Sex',data=train)

sb.countplot(x='Survived',hue='Pclass',data=train)

sb.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)

sb.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4))

# data cleaning - handling null values

sb.boxplot(x='Pclass',y='Age',data=train,palette='winter')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# for cabin column there are more nul values. so feature enginerring is to be done. More time now. so dropping colmn now.

train.drop('Cabin',axis=1,inplace=True)

sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

train.head()

# converting categorical featurews

train.columns

train.info()

pd.get_dummies(train['Embarked'],drop_first=True).head()

embark = pd.get_dummies(train['Embarked'],drop_first=True)

sex = pd.get_dummies(train['Sex'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.info()

train = pd.concat([train,sex,embark],axis=1)

train.info()

'''
Split dataset into train and test
'''
# train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.3,random_state=101)

'''
applying logistic regression model
'''

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predic = logmodel.predict(X_test)

from sklearn.metrics import confusion_matrix

accuracy = confusion_matrix(y_test,predic)

accuracy

from sklearn.metrics import accuracy_score

accu_score = accuracy_score(y_test,predic)

accu_score
# accuracy of 0.77

'''
applying xgboost model
'''

import xgboost as xgb

XGBModel = xgb.XGBClassifier()
XGBModel.fit(X_train,y_train)

XGBPredict = XGBModel.predict(X_test)

XGBScore = accuracy_score(y_test,XGBPredict)
# accuracy of 0.80

'''
applying RandomForestClassifier model
'''

from sklearn.ensemble import RandomForestClassifier

RFCModel = RandomForestClassifier()
RFCModel.fit(X_train,y_train)

RFCPredict = RFCModel.predict(X_test)

RFCScore = accuracy_score(y_test,RFCPredict)
# accuracy of 0.79

from sklearn.model_selection import GridSearchCV

RFCParams = {
        'n_estimators' : [320,330,340],
        'max_depth':[8,9,10,11,12],
        'min_samples_leaf' : [2,4],
        'max_features' : ['auto','sqrt','log2'],
        'criterion' : ['gini','entropy'],
        }

RFCGridCV = GridSearchCV(RFCModel,param_grid=RFCParams,cv=5)
RFCGridCV.fit(X_train,y_train)

RFCGridCV.best_params_

RFCGridCV.best_estimator_

RFCGridCVPredict = RFCGridCV.predict(X_test)

RFCGridCVPredictScore = accuracy_score(y_test,RFCGridCVPredict)



