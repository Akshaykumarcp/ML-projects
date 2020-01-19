# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 06:49:42 2020

@author: Akshay kumar C P
"""

# I've utilised DR. RYAN AHMED code and modified slightly for learning purpose. 

'''
Steps

1. business case - problem statement and procedure
2. terminologies in ML (vocabolories)
3. visualization
4. model training
5. model evaluation
6. model improving
7. model evaluation
8. conclusion

'''

'''
STEP 1: PROBLEM STATEMENT
Predicting if the cancer diagnosis is benign or malignant based on several observations/features

30 features are used, examples:

  - radius (mean of distances from center to points on the perimeter)
  - texture (standard deviation of gray-scale values)
  - perimeter
  - area
  - smoothness (local variation in radius lengths)
  - compactness (perimeter^2 / area - 1.0)
  - concavity (severity of concave portions of the contour)
  - concave points (number of concave portions of the contour)
  - symmetry 
  - fractal dimension ("coastline approximation" - 1)
Datasets are linearly separable using all 30 input features

Number of Instances: 569

Class Distribution: 212 Malignant, 357 Benign

Target class:

   - Malignant (0)
   - Benign (1)
   
Refer below link for more info about the dataset 

https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

'''
# import libraries

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

# import dataset

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

# let's get to know more about the imported data
print('raw dataset')
print(cancer)

print('Imported dataset type')
print(type(cancer))

print('no of rows and columns')
print(cancer['data'].shape)

print('dictionary keys from the dataset')
print(cancer.keys())

print('Description of the column DESCR')
print(cancer['DESCR'])

print('Target attribute')
print(cancer['target'])
# 0 is malignant 
# 1 is Benign

print('Target Names')
print(cancer['target_names'])

print('Feature/Column names')
print(cancer['feature_names'])

# let's convert dataset from <class 'sklearn.utils.Bunch'> into dataframe

# dataframe can be used to manipulate data well

cancerDataframe = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns=np.append(cancer['feature_names'],['target']))

print('Top 5 rows')
print(cancerDataframe.head())

print('Bottom 5 rows')
print(cancerDataframe.tail())

'''
step 3 - visualization

'''

print('Feature/Column names')
print(cancerDataframe.columns)

print('Pairplot for 5 columns')
# hue for specifying target class where in visualization will be effective by different colors
print(sb.pairplot(cancerDataframe,hue='target',vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness']))

print('Countplot for target attribute/feature')
print(sb.countplot(cancerDataframe['target']))

print('Scatterplot for viewing relationship between two attributes/features')
print(sb.scatterplot(x='mean area',y='mean smoothness',hue='target',data=cancerDataframe))

# Let's set fig size (default fig is small so)
plt.figure(figsize=(20,10))

print('Heatmap for viewing co-relation between all attributes/features')
print(sb.heatmap(cancerDataframe.corr(),annot = True))

'''
Step 4 - model training
'''

# split data into dependent and independent attribute/feature

X = cancerDataframe.drop(['target'],axis=1) # independent feature
y = cancerDataframe['target'] # dependent feature
print("Dataset divided into independent and dependent features")

# split data into train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)
print("Dataset divided into train_test_split")
# we're ready to train model

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 

svc_model = SVC()
print("Model created")

svc_model.fit(X_train,y_train)
print("Model trained")

y_predict = svc_model.predict(X_test)

# generalized model - work better.
# evaluate model by confusion matrix.

'''
Step 5 - model evaluation
'''

cm = confusion_matrix(y_test,y_predict)

print("Model evaluation result by confusion matric")
print(cm)
print(sb.heatmap(cm,annot=True))
print('BAd results so lets improvise')

# We got bad results hence let's improve the model

'''
step 6 - model improvisation
'''

# improving the model by

# 1. normalising data and
# 2. SVM parameter optimization

# 1. perform normalization (normalising data)

# for training dataset

print("improving model via normalization begin")

min_train = X_train.min()

range_train = (X_train-min_train).max()

X_train_scaled = (X_train - min_train)/range_train

sb.scatterplot(x = X_train['mean area'], y = X_train['mean smoothness'], hue = y_train)

sb.scatterplot(x = X_train_scaled['mean area'], y = X_train['mean smoothness'], hue = y_train)

# for test dataset

min_test = X_test.min()

range_test = (X_test-min_test).max()

X_test_scaled = (X_test - min_test)/range_test

svc_model.fit(X_train_scaled,y_train)

y_predict = svc_model.predict(X_test_scaled)

print('Model improved using normalization')

cm = confusion_matrix(y_test,y_predict)
print("Model evaluation result by confusion matric")
print(cm)
print(sb.heatmap(cm,annot=True))

print("classfication report")
print(classification_report(y_test,y_predict))

print("improving model via normalization end")

# done with normalization

# 2. SVM parameter optimization 

print("improving model via parameter optimization begin")

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']} 

from sklearn.model_selection import GridSearchCV

# verbose - how many value need to display while  searching for the grid

grid = GridSearchCV(SVC(), param_grid,refit = True, verbose = 4)

grid.fit(X_train_scaled,y_train)

grid.best_params_

grid.best_estimator_

grid_predic = grid.predict(X_test_scaled)

print("improving model via parameter optimization end")

'''
step 7 - model evaluation
'''
cm = confusion_matrix(y_test,grid_predic)
print("Model evaluation result by confusion matric")
print(cm)
print(sb.heatmap(cm,annot=True))
print("classfication report")
print(classification_report(y_test,grid_predic))

'''
step 8 - conclusion
'''
print('97 % accuracy of prediction')
