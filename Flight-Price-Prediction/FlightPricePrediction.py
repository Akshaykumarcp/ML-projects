# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 10:00:29 2020

@author: Akshay kumar C P
"""
# handle 
# date column
# 1 column value having multiple meaning

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

df_train = pd.read_excel('Data_Train.xlsx')
df_test = pd.read_excel('Test_set.xlsx')

df_train.head()
df_test.head()

# Let's combine train and test for creating a generalized model

df_complete = df_train.append(df_test,sort=False)

# df_test has nan has values for the price feature

'''
feature engineering
'''

# check datatypes

df_complete.dtypes

# if date is in above format i,e 01/01/2020, we need to split it into date , month and year - and drop date feature

df_complete['Date'] = df_complete['Date_of_Journey'].str.split('/').str[0]

df_complete['Month'] = df_complete['Date_of_Journey'].str.split('/').str[1]

df_complete['Year'] = df_complete['Date_of_Journey'].str.split('/').str[2]

df_complete.dtypes

# the above created features are in object datatypes. Need to convert into int

df_complete['Date'] = df_complete['Date'].astype(int)
df_complete['Month'] = df_complete['Month'].astype(int)
df_complete['Year'] = df_complete['Year'].astype(int)

df_complete.dtypes

# drop date_of_journey column

df_complete = df_complete.drop(['Date_of_Journey'],axis=1)

# same approach for "arrival_time" feature

df_complete['Arrival_Time'] = df_complete['Arrival_Time'].str.split(' ').str[0]

# feature - total_stops - categorical feature
# check null values

df_complete['Total_Stops'].isnull().sum() # only 1 row has total_stops as null

# lets c which row

df_complete[df_complete['Total_Stops'].isnull()] 

# handling it - only 1 nan so lets replace it with 1 stop

df_complete['Total_Stops']=df_complete['Total_Stops'].fillna('1 stop')

df_complete[df_complete['Total_Stops'].isnull()] # no rows 

# non-stop as a value in total_stops meaning that  there is no stops i,e 0 stops. ;ets replace to 0
# label encoding also can be applied. but here we're doing programic value

df_complete['Total_Stops']=df_complete['Total_Stops'].replace('non-stop','0 stop')

# let's split and retrive only int values

df_complete['Stop'] = df_complete['Total_Stops'].str.split(' ').str[0]

# convert object type of stop feature to int

df_complete['Stop'] = df_complete['Stop'].astype(int)

# drop  total_stops

df_complete = df_complete.drop(['Total_Stops'],axis=1)

# lets seperate hours and minutes from arrival_time feature

df_complete['Arrival_Hour'] = df_complete['Arrival_Time'].str.split(':').str[0]
df_complete['Arrival_Minute'] = df_complete['Arrival_Time'].str.split(':').str[1]

# convert object type of Arrival_Hour and  Arrival_Minute feature to int

df_complete['Arrival_Hour'] = df_complete['Arrival_Hour'].astype(int)
df_complete['Arrival_Minute'] = df_complete['Arrival_Minute'].astype(int)

# drop arrival_time feature

df_complete = df_complete.drop(['Arrival_Time'],axis=1)


# lets seperate hours and minutes from Dep_Time  feature

df_complete['Dep_Hour'] = df_complete['Dep_Time'].str.split(':').str[0]
df_complete['Dep_Minute'] = df_complete['Dep_Time'].str.split(':').str[1]

# convert object type of Arrival_Hour and  Arrival_Minute feature to int

df_complete['Dep_Hour'] = df_complete['Dep_Hour'].astype(int)
df_complete['Dep_Minute'] = df_complete['Dep_Minute'].astype(int)

# drop arrival_time feature

df_complete = df_complete.drop(['Dep_Time'],axis=1)

# route - feature - split based on arrow

df_complete['Route 1']= df_complete['Route'].str.split('→').str[0]
df_complete['Route 2']= df_complete['Route'].str.split('→').str[1]
df_complete['Route 3']= df_complete['Route'].str.split('→').str[2]
df_complete['Route 4']= df_complete['Route'].str.split('→').str[3]
df_complete['Route 5']= df_complete['Route'].str.split('→').str[4]

# handle nan values for above split 

df_complete['Route 1'].fillna("None",inplace=True)
df_complete['Route 2'].fillna("None",inplace=True)
df_complete['Route 3'].fillna("None",inplace=True)
df_complete['Route 4'].fillna("None",inplace=True)
df_complete['Route 5'].fillna("None",inplace=True)

# handle nan values for price features

df_complete['Price'].isnull().sum()

df_complete['Price'].fillna((df_complete['Price'].mean()),inplace=True)

df_complete=df_complete.drop(['Route'],axis=1)
df_complete=df_complete.drop(['Duration'],axis=1)

# handle missing values

df_complete.isnull().sum()
# no missing values

# apply label encoding for categorical features

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df_complete['Airline'] = encoder.fit_transform(df_complete['Airline'])
df_complete['Source'] = encoder.fit_transform(df_complete['Source'])
df_complete['Destination'] = encoder.fit_transform(df_complete['Destination'])
df_complete['Additional_Info'] = encoder.fit_transform(df_complete['Additional_Info'])
df_complete['Route 1'] = encoder.fit_transform(df_complete['Route 1'])
df_complete['Route 2'] = encoder.fit_transform(df_complete['Route 2'])
df_complete['Route 3'] = encoder.fit_transform(df_complete['Route 3'])
df_complete['Route 4'] = encoder.fit_transform(df_complete['Route 4'])
df_complete['Route 5'] = encoder.fit_transform(df_complete['Route 5'])

'''
feature selection
'''

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel

# split into train and test after doing feature engineering

df_train2 = df_complete[0:10683]
df_test2 = df_complete[10683:]

X = df_train2.drop(['Price'],axis=1)
y= df_train2.Price

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

# apply feature section on X_train

# lasso has a penalty paramter i,e more feature , it penalizes features. if alpha values is greate les features will be seleceted

# initialze the ovbject
model = SelectFromModel(Lasso(alpha=0.005, random_state=0))

model.fit(X_train,y_train)

model.get_support()

selected_features = X_train.columns[(model.get_support())]

# year feature has weaker cor so let's drop

X_train = X_train.drop(['Year'],axis=1)

X_test = X_test.drop(['Year'],axis=1)

# model building linear regression, dicision tree regression  , XGBoost etc


