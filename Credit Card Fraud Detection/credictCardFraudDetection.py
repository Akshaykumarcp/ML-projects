# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:56:27 2020

@author: APadashetti
"""
'''

import lib's and dataset
'''
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
#from pandas_profiling import ProfileReport

df = pd.read_csv('creditcard.csv')

'''
More info about dataset
'''

df.info()

dfDesc = df.describe()

df.dtypes

df.columns

df['Class'].unique()

df['V1'].value_counts()
df['V2'].value_counts()
df['V3'].value_counts()
df['V4'].value_counts()
df['V5'].value_counts()
df['V6'].value_counts()
df['V7'].value_counts()
df['V8'].value_counts()
df['V9'].value_counts()
df['V10'].value_counts()
df['V11'].value_counts()
df['V12'].value_counts()
df['V13'].value_counts()
df['V14'].value_counts()
df['V15'].value_counts()
df['V16'].value_counts()
df['V17'].value_counts()
df['V18'].value_counts()
df['V19'].value_counts()
df['V20'].value_counts()
df['V21'].value_counts()
df['V22'].value_counts()
df['V23'].value_counts()
df['V24'].value_counts()
df['V25'].value_counts()
df['V26'].value_counts()
df['V27'].value_counts()
df['V28'].value_counts()
df['Time'].value_counts()
df['Amount'].value_counts()


# Let's see target feature counts - way 1
df['Class'].value_counts()
# 0 - 284315
# 1 - 492

# Let's see target feature counts - way 2
from collections import Counter
Counter(df['Class'])

# dataset is imbalanced

df.isnull().sum().max()

# The classes are heavily skewed we need to solve this issue later.
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')




'''
visualization
'''

sb.pairplot(df)

sb.countplot(df['Class'])

#sb.pairplot(df,)

# heatmap - way 1 
plt.figure(figsize=(40,30))
sb.heatmap(df.corr(),annot=True)

# heatmap - way 2- better at visualization 
corr = df.corr()
top_corr_features = corr.index

plt.figure(figsize=(40,30))
sb.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# let's c how data columns are distributed in dataset

sb.distplot(df['V1'])
sb.distplot(df['V2'])
sb.distplot(df['V3'])
sb.distplot(df['V4'])
sb.distplot(df['V5'])
sb.distplot(df['V6'])
sb.distplot(df['V7'])
sb.distplot(df['V8'])
sb.distplot(df['V9'])
sb.distplot(df['V10'])
sb.distplot(df['V11'])
sb.distplot(df['V12'])
sb.distplot(df['V13'])
sb.distplot(df['V14'])
sb.distplot(df['V15'])
sb.distplot(df['V16'])
sb.distplot(df['V17'])
sb.distplot(df['V18'])
sb.distplot(df['V19'])
sb.distplot(df['V20'])
sb.distplot(df['V21'])
sb.distplot(df['V22'])
sb.distplot(df['V23'])
sb.distplot(df['V24'])
sb.distplot(df['V25'])
sb.distplot(df['V26'])
sb.distplot(df['V27'])
sb.distplot(df['V28'])
sb.distplot(df['Time'])
sb.distplot(df['Amount'])
sb.distplot(df['Class'])

'''
feature engineering
'''

df.isnull().sum()

'''
data split into independent and dependent features
'''

df_copy = df.copy()



X = df_copy.iloc[:,:-1]
y = df_copy.iloc[:,30]

'''
seperating fraud and normal transaction
'''

# get fraud and normal dataset

fraud = df_copy[df_copy['Class']==1]

normal = df_copy[df_copy['Class']==0]

fraud['Class'].value_counts()
normal['Class'].value_counts()

# Fraud transactions heatmap 
Fraudcorr = fraud.corr()
Fraudcorrfeatures = Fraudcorr.index

plt.figure(figsize=(40,30))
sb.heatmap(df[Fraudcorrfeatures].corr(),annot=True,cmap="RdYlGn")

print(fraud.shape,normal.shape)

'''
under sampling dataset using NearMiss
'''

from imblearn.under_sampling import NearMiss

nm = NearMiss()
X_res,y_res = nm.fit_sample(X,y)

'''
over sampling dataset using RandomOverSampler
'''

from imblearn.over_sampling import RandomOverSampler

ROS = RandomOverSampler()

X_train_OS, y_test_OS = ROS.fit_sample(X,y)

print(X_train_OS.shape,y_test_OS.shape)

'''
split data into train and test set
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_OS,y_test_OS,test_size=0.3,random_state=20)

'''
Build Model and evaluate
'''

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


'''
Decision Tree classifier 
'''

DTCModel = DecisionTreeClassifier()
DTCModel.fit(X_train,y_train)
DTCPredict = DTCModel.predict(X_test)

DTCScore = accuracy_score(y_test,DTCPredict)
print(classification_report(y_test,DTCPredict))
print(confusion_matrix(y_test,DTCPredict))

'''
under sampleing accuracy - 0.959
over sampleing accuracy - 0.999 - is the best model. Usually tree based models are prefered for imbalanced dataset

[[148   4]
 [  8 136]]

'''


'''
KNN classifier
'''

KNCModel = KNeighborsClassifier()
KNCModel.fit(X_train,y_train)
KNCPredict = KNCModel.predict(X_test)

KNCScore = accuracy_score(y_test,KNCPredict)
print(classification_report(y_test,KNCPredict))
print(confusion_matrix(y_test,KNCPredict))

'''

under sampleing accuracy - 0.912
over sampleing accuracy - 0.998

[[149   3]
 [ 23 121]]

'''

'''
Gaussian NB
'''

GNBModel = GaussianNB()
GNBModel.fit(X_train,y_train)
GNBPredict = GNBModel.predict(X_test)

GNBScore = accuracy_score(y_test,GNBPredict)
print(classification_report(y_test,GNBPredict))
print(confusion_matrix(y_test,GNBPredict))

'''

under sampleing accuracy - 0.898
over sampleing accuracy - 0.865

[[151   1]
 [ 29 115]]

'''


'''
Logistic Regression
'''
LRModel = LogisticRegression()
LRModel.fit(X_train,y_train)
LRPredict = LRModel.predict(X_test)

LRScore = accuracy_score(y_test,LRPredict)
print(classification_report(y_test,LRPredict))
print(confusion_matrix(y_test,LRPredict))

'''

under sampleing accuracy - 0.956
over sampleing accuracy - 0.936

[[149   3]
 [ 10 134]]

'''