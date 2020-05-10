# -*- coding: utf-8 -*-
"""
Created on Thu May  7 05:49:48 2020

@author: Akshay kumar C P
"""

'''

Attributes: 
    
Id: listing id
url: listing URL
region: craigslist region
region_url: region URL
price: rent per month (Target Column)
type: housing type
sqfeet: total square footage
beds:number of beds
baths:number of bathrooms
cats_allowed: cats allowed boolean (1 = yes, 0 = no)
dogs_allowed: dogs allowed boolean
smoking_allowed: smoking allowed boolean
wheelchair_access: has wheelchair access boolean
electric_vehicle_charge: has electric vehicle charger boolean
comes_furnished: comes with furniture boolean
laundry_options: laundry options available
parking_options: parking options available
image_url: image URL
description: description by poster
lat: latitude
long: longitude
state: state of listing


The Problem Statement: 
    
To build an application which predicts the monthly rental of a house based 
on the given attributes.

'''

''' IMPORT LIB'S' '''

import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt

''' IMPORT DATASET '''
house_dataset_original = pd.read_csv("housing_train.csv")

#copy of dataset to make modification
house_dataset = house_dataset_original.copy()

house_dataset.info()

house_dataset.describe()

house_dataset.count()

house_dataset.columns

house_dataset.corr()

''' HANDLING NULL VALUES '''

house_dataset.isnull().sum()

'''
below are the null values count :

laundry_options            54311
parking_options            95135
description                    2
lat                         1419
long                        1419
state                          1
'''

# let's see null values in percentage representation

house_dataset.isnull().sum() / len(house_dataset) * 100

# let's replace state feature, 1 null values by mode of 'state' column
house_dataset['state'].mode() # ca is the mode value, so let's replace 

# display row of state which is having null value
house_dataset[house_dataset['state'].isnull()]

# let's replace state feature, 1 null value with mode of its column
house_dataset['state'].fillna("ca",inplace=True) # success

# display row of description which is having null value
description_null_rows = house_dataset[house_dataset['description'].isnull()] # 2 rows are having NAN for description column

# so based on existing region and respective description we shall replace to NAN rows

rows_with_region_hudson = house_dataset[house_dataset['url']=='https://hudsonvalley.craigslist.org/apa/d/poughkeepsie-2-bedroom-duplex-in-quiet/7040721424.html']

#drop description as there is no similar row based on url/region to impute for description

house_dataset = house_dataset.drop([115045,247410]) # dropped null rows of description features

'''
handle lat and long null values

lat                         0.535092 %
long                        0.535092 %
'''

# display row of lat and long which is having null value
lat_null_rows = house_dataset[house_dataset['lat'].isnull()]

#house_dataset = house_dataset.dropna(subset[lat,long]) dint work 

'''
handle lat and long null values

laundry_options            54310
parking_options            95134
'''

# display row of lat and long which is having null value
laundry_options_null_rows = house_dataset[house_dataset['laundry_options'].isnull()]

# let's try the accuuracy of the model after dropping all the null valued rows

house_dataset = house_dataset.dropna()

#house_dataset['price'].value_counts()

#sb.countplot(x='price',data=house_dataset,)

# drop unwanted columns

house_dataset = house_dataset.drop(columns=['id','region_url','region_url','url','image_url','description','lat','long'])

# unique row value counts for all columns
for i in house_dataset.columns:
    print(house_dataset[i].value_counts())
    
# categorical features
categorical_features = house_dataset.select_dtypes(include=['object']).columns

house_dataset[categorical_features]

''' DATA PREPROCESSING '''

from sklearn.preprocessing import LabelEncoder, normalize

le = LabelEncoder()

house_dataset['region'] = le.fit_transform(house_dataset['region'])
house_dataset['type'] = le.fit_transform(house_dataset['type'])
house_dataset['laundry_options'] = le.fit_transform(house_dataset['laundry_options'])
house_dataset['parking_options'] = le.fit_transform(house_dataset['parking_options'])
house_dataset['state'] = le.fit_transform(house_dataset['state'])

''' VISUALIZATION '''

# PAIR PLOT
# sb.pairplot(house_dataset) # costly operation

# CORRELATION MATRIX
#get correlations of each features in dataset
corrmat = house_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sb.heatmap(house_dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

corrmat.index

#

sb.distplot(house_dataset_original['price'],)


''' SPLIT DATA INTO X and y '''

y = house_dataset.iloc[:,1]
house_dataset = house_dataset.drop(['price'],axis=1)
X = house_dataset.iloc[:,0:-1]

''' FEATURE NORMALIZATION '''
# X_normalized = normalize(X)

# split data into train test split

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

''' FEATURE IMPORTANCE '''

# costly operation
from sklearn.ensemble import ExtraTreesRegressor

ETRModel = ExtraTreesRegressor()
ETRModel.fit(X,y)

X.head()
ETRModel.feature_importances_

# plot feature's that are important

feature_importances = pd.Series(ETRModel.feature_importances_)
feature_importances.nlargest(6),plt.plot(kind='barh')
plt.show()

''' APPLY MODELS '''

# apply linear regression model

from sklearn.linear_model import LinearRegression

LRModel = LinearRegression()
LRModel.fit(X_train,y_train)

LRModel.coef_
LRModel.intercept_

print("Coefficient of determination R^2 for LinearRegression <-- on train set: {}".format(LRModel.score(X_train, y_train)))
print("Coefficient of determination R^2 for LinearRegression <-- on train set: {}".format(LRModel.score(X_test, y_test)))

coeff_df = pd.DataFrame(LRModel.coef_,X.columns,columns=['Coefficient'])
coeff_df

LRModelPredict = LRModel.predict(X_test)

sb.distplot(y_test-LRModelPredict)
sb.scatterplot(x=y_test,y=LRModelPredict)

# APPLY KNEIGHORSREGRESSOR

from sklearn.neighbors import KNeighborsRegressor

KNRModel = KNeighborsRegressor()
KNRModel.fit(X_train,y_train)

print("Coefficient of determination R^2 for KNeighborsRegressor <-- on train set: {}".format(KNRModel.score(X_train, y_train)))
print("Coefficient of determination R^2 for KNeighborsRegressor <-- on train set: {}".format(KNRModel.score(X_test, y_test)))

KNRPredict = KNRModel.predict(X_test)

sb.distplot(y_test-KNRPredict)
sb.scatterplot(x=y_test,y=KNRPredict)

# APPLY DECISIONTREEREGRESSOR

from sklearn.tree import DecisionTreeRegressor

DTRModel = DecisionTreeRegressor()
DTRModel.fit(X_train,y_train)

print("Coefficient of determination R^2 for DecisionTreeRegressor <-- on train set: {}".format(DTRModel.score(X_train, y_train)))
print("Coefficient of determination R^2 for DecisionTreeRegressor <-- on train set: {}".format(DTRModel.score(X_test, y_test)))

DTRPredict = DTRModel.predict(X_test)

sb.distplot(y_test-DTRPredict)
sb.scatterplot(x=y_test,y=DTRPredict)

# APPLY RANDOMFORESTREGRESSOR MODEL

from sklearn.ensemble import RandomForestRegressor

RFRModel = RandomForestRegressor()
RFRModel.fit(X_train,y_train)

print("Coefficient of determination R^2 for RandomForestRegressor<-- on train set: {}".format(RFRModel.score(X_train, y_train)))
print("Coefficient of determination R^2 for RandomForestRegressor <-- on train set: {}".format(RFRModel.score(X_test, y_test)))

RFRPredict = RFRModel.predict(X_test)

sb.distplot(y_test-RFRPredict)
sb.scatterplot(x=y_test,y=RFRPredict)

# apply XGBOOST MODEL

import xgboost
XGBoostModel = xgboost.XGBRegressor()
XGBoostModel.fit(X_train,y_train)

XGBoostPredict = XGBoostModel.predict(X_test)

sb.distplot(y_test-XGBoostPredict)
sb.scatterplot(x=y_test,y=XGBoostPredict)


''' CROSS VALIDATION '''

from sklearn.model_selection import cross_val_score

# CV for Linear regression
LRCVScore = cross_val_score(LRModel,X,y,cv=5)
LinearRegressorCVScoreMean = LRCVScore.mean()

# CV for KNEIGHORSREGRESSOR
KNNCVScore = cross_val_score(KNRModel,X,y,cv=5)
KNNScoreMean = KNNCVScore.mean()

# CV for DECISIONTREEREGRESSOR
DTRModelCVScore = cross_val_score(DTRModel,X,y,cv=5)
DTRModelScoreMean = DTRModelCVScore.mean()

# CV for RANDOMFORESTREGRESSOR
RFRModelCVScore = cross_val_score(RFRModel,X,y,cv=5)
RFRModelScoreMean = RFRModelCVScore.mean()

# CV for XGBOOST
XGBoostModelCVScore = cross_val_score(XGBoostModel,X,y,cv=5)
XGBoostModelScoreMean = XGBoostModelCVScore.mean()


# model evaluation

from sklearn import metrics

# METRIC EVALUATION FOR LINEARREGRESSOR
print("------LINEARREGRESSOR Evaluation metrics------")
print('r2_score:', metrics.r2_score(y_test, LRModelPredict))
print('MAE:', metrics.mean_absolute_error(y_test, LRModelPredict))
print('MSE:', metrics.mean_squared_error(y_test, LRModelPredict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, LRModelPredict)))

# METRIC EVALUATION FOR KNEIGHORSREGRESSOR

print("------KNEIGHORSREGRESSOR Evaluation metrics------")
print('r2_score:', metrics.r2_score(y_test, KNRPredict))
print('MAE:', metrics.mean_absolute_error(y_test, KNRPredict))
print('MSE:', metrics.mean_squared_error(y_test, KNRPredict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, KNRPredict)))

# METRIC EVALUATION FOR DECISIONTREEREGRESSOR

print("------DECISIONTREEREGRESSOR Evaluation metrics------")
print('r2_score:', metrics.r2_score(y_test, DTRPredict))
print('MAE:', metrics.mean_absolute_error(y_test, DTRPredict))
print('MSE:', metrics.mean_squared_error(y_test, DTRPredict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, DTRPredict)))

# METRIC EVALUATION FOR RANDOMFORESTREGRESSOR

print("------RANDOMFORESTREGRESSOR Evaluation metrics------")
print('r2_score:', metrics.r2_score(y_test, RFRPredict))
print('MAE:', metrics.mean_absolute_error(y_test, RFRPredict))
print('MSE:', metrics.mean_squared_error(y_test, RFRPredict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, RFRPredict)))

# METRIC EVALUATION FOR XGBOOST

print("------XGBOOST Evaluation metrics------")
print('r2_score:', metrics.r2_score(y_test, XGBoostPredict))
print('MAE:', metrics.mean_absolute_error(y_test, XGBoostPredict))
print('MSE:', metrics.mean_squared_error(y_test, XGBoostPredict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, XGBoostPredict)))







