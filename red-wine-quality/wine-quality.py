# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 05:30:11 2020

@author: Akshay kumar C P
"""

'''
import lib's and dataset
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

#df_kaggle  = pd.read_csv('dataset-from-kaggle/winequality-red.csv')

df_uci_red = pd.read_csv('dataset-from-uci.edu/winequality-red.csv',sep=';')
df_uci_white = pd.read_csv('dataset-from-uci.edu/winequality-white.csv',sep=';')

df = pd.concat([df_uci_red,df_uci_white])

'''
About Dataset
'''

df.columns

df.info()

df['quality'].unique()

df['quality'].value_counts()

des = df.describe()

'''
fixed acidity - 25% - 7.1 and 50% - 7.9. Not much of a variance. Could explain the huge number of outliers
volatile acididty - similar reasoning
citric acid - seems to be somewhat uniformly distributed
residual sugar - min - 0.9, max - 15!! Waaaaay too much difference. Could explain the outliers.
chlorides - same as residual sugar. Min - 0.012, max - 0.611
free sulfur dioxide, total suflur dioxide - same explanation as above
'''

df.corr()

# Let's c how target feature is distributed.

sb.distplot(df.iloc[:,11])

# count of each target variable

from collections import Counter
Counter(df['quality'])

'''
Visualization
'''

sb.pairplot(df,hue='quality')

sb.countplot(df['quality'])

# Heat map way 1

plt.figure(figsize=(20,30))
sb.heatmap(df.corr(),annot=True)

# Heat map way 2
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sb.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

corrmat.index

# let's c how data columns are distributed in dataset

#fig = plt.figure(figsize = (10,6))
# fixed acidity dint give any specification
sb.barplot(x = 'quality', y = 'fixed acidity', data = df)

#Here we see that its quite a downing trend in the volatile acidity as we go higher the quality 
sb.barplot(x = 'quality', y = 'chlorides', data = df)

#Composition of citric acid go higher as we go higher in the quality of the wine
sb.barplot(x = 'quality', y = 'citric acid', data = df)

sb.barplot(x = 'quality', y = 'residual sugar', data = df)

#Composition of chloride also go down as we go higher in the quality of the wine
sb.barplot(x = 'quality', y = 'chlorides', data = df)

sb.barplot(x = 'quality', y = 'free sulfur dioxide', data = df)

sb.barplot(x = 'quality', y = 'total sulfur dioxide', data = df)

#Sulphates level goes higher with the quality of wine
sb.barplot(x = 'quality', y = 'sulphates', data = df)

#Alcohol level also goes higher as te quality of wine increases
sb.barplot(x = 'quality', y = 'alcohol', data = df)


'''
Feature Engineering
'''

df.isnull().sum()
# no nan values

'''
data split into independent and dependent features
'''

'''
first

df_copy = df.copy()

X = df_copy.iloc[:,:-1]
y = df_copy.iloc[:,11]
'''

df_copy = df.copy()

df_copy.columns

df_copy = df_copy.drop(['fixed acidity','citric acid','chlorides'],axis=1)


reviews = []
for i in df_copy['quality']:
    if i >= 1 and i <= 3:
        reviews.append('1')
    elif i >= 4 and i <= 7:
        reviews.append('2')
    elif i >= 8 and i <= 10:
        reviews.append('3')
df_copy['Reviews'] = reviews

df_copy = df_copy.drop(['quality'],axis=1)

X = df_copy.iloc[:,:-1]
y = df_copy.iloc[:,8]

y.value_counts()

'''
feature importance
'''

from sklearn.ensemble import ExtraTreesRegressor

ETRModel = ExtraTreesRegressor()
ETRModel.fit(X,y)

print(ETRModel.feature_importances_)

ImpFeatures = pd.Series(ETRModel.feature_importances_,index=X.columns)
ImpFeatures.nlargest(7).plot(kind='barh',)


'''
normalization
'''

from sklearn.preprocessing import StandardScaler

SC = StandardScaler()

X = SC.fit_transform(X)

'''
split data into train and test set
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

'''
Build Model and evaluate
'''
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

DTCModel = DecisionTreeClassifier()
DTCModel.fit(X_train,y_train)
DTCPredict = DTCModel.predict(X_test)

DTCScore = accuracy_score(y_test,DTCPredict)
print(cross_val_score(DTCModel, X, y, cv=10, scoring ='accuracy').mean())

RFCModel = RandomForestClassifier()
RFCModel.fit(X_train,y_train)
RFCPredict = RFCModel.predict(X_test)

RFCScore = accuracy_score(y_test,RFCPredict)
print(cross_val_score(RFCModel, X, y, cv=10, scoring ='accuracy').mean())

KNCModel = KNeighborsClassifier()
KNCModel.fit(X_train,y_train)
KNCPredict = KNCModel.predict(X_test)

KNCScore = accuracy_score(y_test,KNCPredict)
print(cross_val_score(KNCModel, X, y, cv=10, scoring ='accuracy').mean())

GNBModel = GaussianNB()
GNBModel.fit(X_train,y_train)
GNBPredict = GNBModel.predict(X_test)

GNBScore = accuracy_score(y_test,GNBPredict)
print(cross_val_score(GNBModel, X, y, cv=10, scoring ='accuracy').mean())

SVCModel = SVC()
SVCModel.fit(X_train,y_train)
SVCPredict = SVCModel.predict(X_test)

SVCScore = accuracy_score(y_test,SVCPredict)
print(cross_val_score(SVCModel, X, y, cv=10, scoring ='accuracy').mean())

XGBModel = xgb.XGBClassifier()
XGBModel.fit(X_train,y_train)
XGBPredict = XGBModel.predict(X_test)

XGBScore = accuracy_score(y_test,XGBPredict)
print(cross_val_score(XGBModel, X, y, cv=10, scoring ='accuracy').mean())



'''
hyper parameter tuning for SVC
'''

SVCParam = {
        'C' : [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
        'kernel' : ['linear','rbf'],
        'gamma' : [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
        }

SVCGS = GridSearchCV(SVCModel,param_grid=SVCParam,scoring='accuracy',cv=10)

SVCGS.fit(X_train,y_train)

SVCGS.best_params_

SVCGS.best_estimator_

SVCGS.fit(X_train,y_train)

SVCGSPredict = SVCGS.predict(X_test)

SVCGSPredictScore = accuracy_score(y_test,SVCGSPredict)
print(classification_report(y_test, SVCGSPredict))

'''
Pick the algo which is performing good and have fun!
'''


'''
Regression Model
'''

from sklearn.tree import DecisionTreeRegressor

DTRModel = DecisionTreeRegressor()

DTRModel.fit(X_train,y_train)

print('Co-efficient of determination R^2 on train set : {}'.format(DTRModel.score(X_train,y_train)))
print('Co-efficient of determination R^2 on test set : {}'.format(DTRModel.score(X_test,y_test)))


CVScore = cross_val_score(DTRModel,X,y,cv=5)
SCSoreMean = CVScore.mean()

DTRPredict = DTRModel.predict(X_test)

sb.distplot(DTRPredict)

sb.scatterplot(y_test,DTRPredict) # seaborn
plt.scatter(y_test,DTRPredict) # matplotlib

# first = -0.038905236798634446

r2scoree = r2_score(y_test,DTRPredict)


# first
# MAE: 0.538974358974359
#MSE: 0.8056410256410257
#RMSE: 0.897575080782118
print('MAE:',metrics.mean_absolute_error(y_test,DTRPredict))
print('MSE:',metrics.mean_squared_error(y_test,DTRPredict))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,DTRPredict)))

'''
hyper parameter tuning for regressor algo
'''

params = {
        "splitter" : ["best","random"],
        "max_depth" : [3,4,5,6,8,10,12,15],
        "min_samples_leaf" : [1,2,3,4,5],
        "min_weight_fraction_leaf" : [0.1,0.2,0.3,0.4],
        "max_features" : ["auto","log2","sqrt",None],
        "max_leaf_nodes" : [None,10,20,30,40,50,60,70]
        }


GSCV = GridSearchCV(DTRModel,param_grid=params,scoring='neg_mean_squared_error',n_jobs=-1,cv=10,verbose=3)
GSCV.fit(X,y)
GSCV.best_params_
GSCV.best_score_
GSCVPredict = GSCV.predict(X_test)

r2scoreWithGSCV = r2_score(y_test,GSCVPredict)

print('MAE:',metrics.mean_absolute_error(y_test,GSCVPredict))
print('MSE:',metrics.mean_squared_error(y_test,GSCVPredict))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,GSCVPredict)))