# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 10:12:56 2020

@author: APadashetti
"""

# import lib's and dataset's

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv('heart-disease-uci/heart.csv')

'''
about data
'''

df.dtypes

df.info()

df_describe = df.describe()

df_columns =df.columns

df_corr = df.corr()

'''
visualization
'''

sb.pairplot(df,hue='target')

sb.countplot(x=df['target'])

# sb.scatterplot(y=df['target'],x=df['age'])

plt.figure(figsize=(40,60))
sb.heatmap(df,annot=True)

# heatmap - way 2- better at visualization 
corr = df.corr()
top_corr_features = corr.index

plt.figure(figsize=(40,30))
sb.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")

# distribution of data
sb.distplot(df['age'])

sb.jointplot(data=df,x='target',y='age')

sb.barplot(x='target',y='age',hue='target',data=df)

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')

plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red")
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.show()


pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(15,6),color=['#DAF7A6','#FF5733' ])
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.show()

pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(15,6),color=['#FFC300','#581845' ])
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.show()

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ])
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.show()

# check null values 
df.isnull().sum()

# no null values found

df_copy = df.copy()

'''
split dataset into dependent and independent features
'''

X = df_copy.iloc[:,:-1]
y = df_copy.iloc[:,13]

'''
feature importance
'''

from sklearn.ensemble import ExtraTreesRegressor
ETRModel = ExtraTreesRegressor()
ETRModel.fit(X,y)

feat_imp = pd.Series(ETRModel.feature_importances_,index=X.columns)
feat_imp.nlargest(7).plot(kind='barh')

'''
Normalization
'''

X = (df_copy - np.min(df_copy)) / (np.max(df_copy) - np.min(df_copy)).values

X= X.iloc[:,:-1]

'''
train and test set
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

'''
model implementation
'''

from sklearn.metrics import classification_report, confusion_matrix
accuracies = {}

#logisticRegression

from sklearn.linear_model import LogisticRegression

LRModel = LogisticRegression()
LRModel.fit(X_train,y_train)

LRPredict = LRModel.predict(X_test)

cm_LR = confusion_matrix(y_test,LRPredict)

classification_report(y_test,LRPredict)

LRScore = LRModel.score(X_test,y_test)

# training accuracy
LRModel.score(X_train,y_train)

# testing accuracy
LRModel.score(X_test,y_test)

accuracies['Logistic Regression'] = LRScore

# KNN model

from sklearn.neighbors import KNeighborsClassifier
KNNModel = KNeighborsClassifier(n_neighbors=3)
KNNModel.fit(X_train,y_train)
KNNPredict = KNNModel.predict(X_test)

cm_KNN = confusion_matrix(y_test,KNNPredict)

classification_report(y_test,KNNPredict)

KNNScore = KNNModel.score(X_test,y_test)
accuracies['KNN'] = KNNScore

# SVM

from sklearn.svm import SVC

SVCModel = SVC(random_state=1)
SVCModel.fit(X_train,y_train)

SVCPredict = SVCModel.predict(X_test)

cm_SVM = confusion_matrix(y_test,SVCPredict)

classification_report(y_test,SVCPredict)

SVCScore = SVCModel.score(X_test,y_test)
accuracies['SVM'] = SVCScore

# NB

from sklearn.naive_bayes import GaussianNB

NBModel = GaussianNB()
NBModel.fit(X_train,y_train)

NBPredict = NBModel.predict(X_test)

cm_NB = confusion_matrix(y_test,NBPredict)

classification_report(y_test,NBPredict)

NBScore = NBModel.score(X_test,y_test)
accuracies['NB']=NBScore

# Decision tree classifier

from sklearn.tree import DecisionTreeClassifier

DTModel = DecisionTreeClassifier()
DTModel.fit(X_train,y_train)

DTPredict = DTModel.predict(X_test)

cm_DTC = confusion_matrix(y_test,DTPredict)

classification_report(y_test,DTPredict)

DTScore = DTModel.score(X_test,y_test)
accuracies['DTC'] = DTScore

# Random forest classifier

from sklearn.ensemble import RandomForestClassifier

RFModel = RandomForestClassifier(n_estimators=1000,random)
RFModel.fit(X_train,y_train)

RFPredict = RFModel.predict(X_test)

cm_RFC = confusion_matrix(y_test,RFPredict)

classification_report(y_test,RFPredict)

RFCScore = RFModel.score(X_test,y_test)
accuracies['RFC'] = RFCScore

# performing hyper parameter tuning for Random forest classifier

random_grid = {
        'n_estimators' : [int(x) for x in np.linspace(start=200,stop=2000,num=10)],
        'max_features' : ['auto','sqrt'],
        'max_depth' : [int(x) for x in np.linspace(10,110,num=11)],
        'min_samples_split' : [2,5,10],
        'min_samples_leaf' : [1,2,4],
        'bootstrap' : [True,False]
        }

print(random_grid)

from sklearn.model_selection import RandomizedSearchCV

RFModel2 = RandomForestClassifier()
RSCVModel = RandomizedSearchCV(estimator=RFModel2,param_distributions=random_grid,n_iter=100,verbose=2,cv=3,random_state=42,n_jobs=-1)
RSCVModel.fit(X_train,y_train)

RSCVModel.best_params_

RSCVModel.best_estimator_

RSCVPredict = RSCVModel.predict(X_test)

cm_RSCV = confusion_matrix(y_test,RSCVPredict)

classification_report(y_test,RSCVPredict)

RFCScore = RSCVModel.score(X_test,y_test)

# not included HPT in plot visualization

# plot all confusion matrix

plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)

plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sb.heatmap(cm_LR,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,2)
plt.title("K Nearest Neighbors Confusion Matrix")
sb.heatmap(cm_KNN,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sb.heatmap(cm_SVM,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sb.heatmap(cm_NB,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,5)
plt.title("Decision Tree Classifier Confusion Matrix")
sb.heatmap(cm_NB,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.subplot(2,3,6)
plt.title("Random Forest Confusion Matrix")
sb.heatmap(cm_RFC,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})

plt.show()

# plot for model comparisons

colors = ["purple", "green", "orange", "magenta","#CFC60E","#0FBBAE"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()

# KNN and Random are giving better accuracy

# when free implement visualization and ROC https://www.kaggle.com/ahmadjaved097/classifying-heart-disease-patients