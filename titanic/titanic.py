# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 07:55:39 2019

@author: Akshay Kumar C P
"""

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')

train.head()

# EDA

# missing data finding

train.columns

train.isnull()

sb.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sb.set_style('whitegrid')
sb.countplot(x='Survived',data=train)

sb.set_style('whitegrid')
sb.countplot(x='Survived',hue='Sex',data=train)

sb.set_style('whitegrid')
sb.countplot(x='Survived',hue='Pclass',data=train)

sb.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=40)

sb.countplot(x='SibSp',data=train)

train['Fare'].hist(color='green',bins=40,figsize=(8,4))

# data cleaning - handling null calues

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

# logistic re

# train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.3,random_state=101)

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

# apply xgboost and random forst R