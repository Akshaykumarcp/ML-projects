# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:07:02 2020

@author: APadashetti
"""

'''
import lib's and dataset
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

df = pd.read_csv('pima-indians-diabetes-database/diabetes.csv')

'''
EDA
'''

df.info()

df_describe = df.describe()

df.dtypes

df.head()

df.columns

df_coor = df.corr()

df['Outcome'].unique()

# Let's check value's and it's count of "outcome" feature

df['Outcome'].value_counts() # - way 1

from collections import Counter 
Counter(df['Outcome'])

#for i in df.columns:
#    print(i," ",df[i].value_counts)
    
'''
visualization
'''

sb.countplot(data=df,x='Outcome',hue='Outcome')

sb.pairplot(df,hue='Outcome')

plt.figure(figsize=(20,20))
sb.heatmap(df.corr(),annot=True,cmap="RdYlGn")

sb.distplot(df['Pregnancies'])
sb.distplot(df['Glucose'])
sb.distplot(df['BloodPressure'])
sb.distplot(df['SkinThickness'])
sb.distplot(df['Insulin'])
sb.distplot(df['BMI'])
sb.distplot(df['DiabetesPedigreeFunction'])
sb.distplot(df['Age'])
sb.distplot(df['Outcome'])

sb.scatterplot(x='Age',y='Outcome',data=df)

sb.barplot(x='Outcome',y='Age',hue='Outcome',data=df)

sb.boxplot(x='Outcome',y='Age',hue='Outcome',data=df)

sb.violinplot(x='Outcome',y='Age',hue='Outcome',data=df)

df.isnull().sum()

'''
independent and dependent features
'''

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

'''
train test split
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)

'''
find zero values and impute mean values
'''

print(" total rows ",len(df))

print("Pregnancies feature total zero values :",len(df.loc[df['Pregnancies'] == 0]))
print("Glucose feature total zero values :",len(df.loc[df['Glucose'] == 0]))
print("BloodPressure feature total zero values :",len(df.loc[df['BloodPressure'] == 0]))
print("SkinThickness feature total zero values :",len(df.loc[df['SkinThickness'] == 0]))
print("Insulin feature total zero values :",len(df.loc[df['Insulin'] == 0]))
print("BMI feature total zero values :",len(df.loc[df['BMI'] == 0]))
print("DiabetesPedigreeFunction feature total zero values :",len(df.loc[df['DiabetesPedigreeFunction'] == 0]))
print("Age feature total zero values :",len(df.loc[df['Age'] == 0]))


X_train_copy = X_train.copy()
X_test_copy = X_test.copy()

from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0,strategy="mean")
X_train_copy = fill_values.fit_transform(X_train)
X_test_copy = fill_values.fit_transform(X_test)

'''
apply model - RFC
'''

from sklearn.ensemble import RandomForestClassifier

RFCModel = RandomForestClassifier(random_state=10)
RFCModel.fit(X_train,y_train) 

RFCPredict = RFCModel.predict(X_test)

'''
model evaluation
'''

from sklearn import metrics

metrics.accuracy_score(y_test,RFCPredict) # 0.74025

'''
apply model - XGBoost
'''

import xgboost

XGBModel = xgboost.XGBClassifier()

XGBModel.fit(X_train,y_train)

XGBPredict = XGBModel.predict(X_test)


'''
model evaluation
'''

metrics.accuracy_score(y_test,XGBPredict) # 0.72077 without hyperparameter optimization

'''
hyper parameter optimization using RandomizedSearchCV
'''

param = {
            "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],
            "max_depth" : [3,4,5,6,8,10,12,15],
            "min_child_weight" : [1,3,5,7],
            "gamma" : [0.0,0.1,0.2,0.3,0.4],
            "colsample_bytree" : [0.3,0.4,0.5,0.7]
        }

from sklearn.model_selection import RandomizedSearchCV

RSCVModel = RandomizedSearchCV(XGBModel,param_distributions=param,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

RSCVModel.fit(X_train,y_train)

RSCVModel.best_estimator_

XGBoostModel2 = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.3, gamma=0.0,
              learning_rate=0.3, max_delta_step=0, max_depth=5,
              min_child_weight=5, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)

from sklearn.model_selection import cross_val_score

CVScore = cross_val_score(XGBoostModel2,X_train,y_train,cv=10)

CVScore.mean() # 0.7588

XGBoostModel2.fit(X_train,y_train)

XGBoostPredict2 = XGBoostModel2.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test,XGBoostPredict2)

XGBoostScore = accuracy_score(y_test,XGBoostPredict2) # 0.6688