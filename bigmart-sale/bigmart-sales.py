# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 05:10:01 2020

@author: Akshay kumar C P
"""

'''
import lib's and dataset
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

print("Successfully imported Libraries")

dataset_train = pd.read_csv('bigmart-sales-data/Train.csv')
dataset_test = pd.read_csv('bigmart-sales-data/Test.csv')

print("Successfully imported DAtaset")


'''
EDA
'''

print(" EDA Begins")

# let's take a copy of dataset such that original dataset doesn't get changed

dataset_trainCopy = dataset_train.copy()
dataset_testCopy = dataset_test.copy()

# Let's look the columns i,e difference between columns

print('Train dataset columns',dataset_trainCopy.columns)
print('Test dataset columns',dataset_testCopy.columns)

# let's know the no of rows and columns in the dataset

print(dataset_trainCopy.shape,dataset_testCopy.shape)

# Let's know the data type of each column

print('Dataset info',dataset_trainCopy.info())

# Let's know the statistical inference of each column

print('Dataset Describe',dataset_trainCopy.describe())

# Let's know how output variable is amoung values

print('Item_Outlet_Sales column describe',dataset_trainCopy['Item_Outlet_Sales'].describe())

# Let's visualize the output variable i,e how data is distributed 

print('Distribution of Item_Outlet_Sales')

sb.distplot(dataset_trainCopy['Item_Outlet_Sales'])

# from the dataset lets know numerical and categorical features

categorical_features = dataset_trainCopy.select_dtypes(include=[np.object])

numerical_features = dataset_trainCopy.select_dtypes(include=[np.number])

print("Categorical Features",categorical_features)
print("Numerical Features",numerical_features)

'''
feature engineering and visualization
'''

# finding missing values for each columns

print('Missing values for each column',dataset_trainCopy.isnull().sum())

# let's visualize using heat map

sb.heatmap(dataset_trainCopy.isnull(),yticklabels=False,cbar=False,cmap='viridis')

# Let's count frequency of categorical variables

for col in categorical_features:
    print('\n %s column' %col)
    print(dataset_trainCopy[col].value_counts())

# lets visualize how outlet size, outlet location type and outlet type is associated with item_oulet_sales
    
sb.boxplot(x='Outlet_Size',y='Item_Outlet_Sales',data=dataset_trainCopy)

sb.boxplot(x='Outlet_Location_Type',y='Item_Outlet_Sales',data=dataset_trainCopy)

sb.boxplot(x='Outlet_Type',y='Item_Outlet_Sales',data=dataset_trainCopy)

sb.boxplot(x='Outlet_Identifier',y='Item_Outlet_Sales',data=dataset_trainCopy,)

sb.boxplot(x='Item_Type',y='Item_Outlet_Sales',data=dataset_trainCopy)

# add average value to item weight null values

# for train data

avg_itemWeightTrain = dataset_trainCopy.pivot_table(values="Item_Weight",index="Item_Identifier")

np.mean(avg_itemWeightTrain)

dataset_trainCopy['new_Item_Weight'] = dataset_trainCopy['Item_Weight'].fillna(12.808553) 

dataset_trainCopy.drop(['Item_Weight'],axis=1,inplace=True)

# for test data

avg_itemWeightTest = dataset_testCopy.pivot_table(values="Item_Weight",index="Item_Identifier")

np.mean(avg_itemWeightTest)

dataset_testCopy['new_Item_Weight'] = dataset_testCopy['Item_Weight'].fillna(12.818091) 

dataset_testCopy.drop(['Item_Weight'],axis=1,inplace=True)

# add average value to item visibility null values

# for train data - item visibility

#dataset_trainCopy['Item_Visibility'] = dataset_trainCopy['Item_Visibility'].map(lambda x : x=0.065683 if x == 0)
'''
for i in dataset_trainCopy['Item_Visibility']:
    if i == 0:
        dataset_trainCopy[i] = 0.065683
'''

avg_itemVisibilityTrain = dataset_trainCopy.pivot_table(values="Item_Visibility",index="Item_Identifier")

mean_Item_VisibilityTrain = np.mean(avg_itemVisibilityTrain)

dataset_trainCopy['Item_Visibility'] = dataset_trainCopy.Item_Visibility.mask(dataset_trainCopy.Item_Visibility == 0,mean_Item_VisibilityTrain)

dataset_trainCopy['New_Item_Visibility'] = dataset_trainCopy['Item_Visibility'].fillna(0.065683)

dataset_trainCopy.drop(['Item_Visibility'],axis=1,inplace=True)

dataset_trainCopy['Outlet_Size'].fillna('Small',inplace=True)
dataset_testCopy['Outlet_Size'].fillna('Small',inplace=True)

'''
for i in range(len(dataset_trainCopy)):
    print(dataset_trainCopy['i'])
'''

'''
def impute_visibility(cols):
    Item_Visibility = cols[0]
    print(Item_Visibility)
    
    if Item_Visibility==0:
        return 0.065683
    else:
        return Item_Visibility
    
IV = dataset_trainCopy[['Item_Visibility']].apply(impute_visibility)
'''
#dataset_trainCopy['new_Item_Visibility'] = dataset_trainCopy[i for i in dataset_trainCopy['Item_Visibility']]

#testing = dataset_trainCopy[i=0.5 for i in dataset_trainCopy['Item_Visibility'] if i == 0]

#dataset_trainCopy.drop(['Item_Visibility'],axis=1,inplace=True)

# for test data

avg_itemVisibilityTest = dataset_testCopy.pivot_table(values="Item_Visibility",index="Item_Identifier")

mean_Item_VisibilityTest = np.mean(avg_itemVisibilityTest)

dataset_testCopy['Item_Visibility'] = dataset_testCopy.Item_Visibility.mask(dataset_testCopy.Item_Visibility == 0,mean_Item_VisibilityTest)

dataset_testCopy['New_Item_Visibility'] = dataset_testCopy['Item_Visibility'].fillna(0.0657868)

dataset_testCopy.drop(['Item_Visibility'],axis=1,inplace=True)

# item visibility is done

# count value counts

# for train
print(dataset_trainCopy['Item_Type'].value_counts())

dataset_trainCopy['Item_Type_Combined'] = dataset_trainCopy['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories:
dataset_trainCopy['Item_Type_Combined'] = dataset_trainCopy['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
dataset_trainCopy['Item_Type_Combined'].value_counts()

# for test

dataset_testCopy['Item_Type_Combined'] = dataset_testCopy['Item_Identifier'].apply(lambda x: x[0:2])

#Rename them to more intuitive categories:
dataset_testCopy['Item_Type_Combined'] = dataset_testCopy['Item_Type_Combined'].map({'FD':'Food',
                                                             'NC':'Non-Consumable',
                                                             'DR':'Drinks'})
dataset_testCopy['Item_Type_Combined'].value_counts()


#dataset_trainCopy = dataset_trainCopy['Outlet_Size'].isnull().isum()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset_trainCopy['Item_Fat_Content'] = le.fit_transform(dataset_trainCopy['Item_Fat_Content'])
dataset_testCopy['Item_Fat_Content'] = le.fit_transform(dataset_testCopy['Item_Fat_Content'])

dataset_trainCopy['Item_Type_Combined'] = le.fit_transform(dataset_trainCopy['Item_Type_Combined'])
dataset_testCopy['Item_Type_Combined'] = le.fit_transform(dataset_testCopy['Item_Type_Combined'])

dataset_trainCopy['Outlet_Location_Type'] = le.fit_transform(dataset_trainCopy['Outlet_Location_Type'])
dataset_testCopy['Outlet_Location_Type'] = le.fit_transform(dataset_testCopy['Outlet_Location_Type'])

dataset_trainCopy['Outlet_Type'] = le.fit_transform(dataset_trainCopy['Outlet_Type'])
dataset_testCopy['Outlet_Type'] = le.fit_transform(dataset_testCopy['Outlet_Type'])

dataset_trainCopy['Outlet_Size'] = le.fit_transform(dataset_trainCopy['Outlet_Size'])
dataset_testCopy['Outlet_Size'] = le.fit_transform(dataset_testCopy['Outlet_Size'])

dataset_trainCopy['Outlet_Identifier'] = le.fit_transform(dataset_trainCopy['Outlet_Identifier'])
dataset_testCopy['Outlet_Identifier'] = le.fit_transform(dataset_testCopy['Outlet_Identifier'])


# drop unwanted columns

dataset_trainCopy.columns

dataset_trainCopy.drop(['Item_Identifier'],axis=1,inplace=True)
dataset_testCopy.drop(['Item_Identifier'],axis=1,inplace=True)

dataset_trainCopy.drop(['Item_Type'],axis=1,inplace=True)
dataset_testCopy.drop(['Item_Type'],axis=1,inplace=True)

#dataset_trainCopy.drop(['Outlet_Identifier'],axis=1,inplace=True)
#dataset_testCopy.drop(['Outlet_Identifier'],axis=1,inplace=True)
'''
dataset_trainCopy.drop(['Outlet_Size'],axis=1,inplace=True)
dataset_testCopy.drop(['Outlet_Size'],axis=1,inplace=True)
'''

'''
dataTest = dataset_trainCopy.copy()


from sklearn.preprocessing import Imputer


impu = Imputer(missing_values="0",strategy="mean")
impu.fit(dataTest['Item_Visibility'])

dataTest['Item_Visibility'] = impu.transform(dataTest['Item_Visibility'])).ravel())

type(dataTest)

'''
'''
sb.pairplot(dataset_train)

dataset_trainCopy.drop(['Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type'],axis=1,inplace=True)
dataset_testCopy.drop(['Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Outlet_Size','Outlet_Type'],axis=1,inplace=True)

dataset_testCopy.isnull().sum()

sb.heatmap(dataset_trainCopy.isnull(),cmap='viridis',yticklabels=False)

#sb.boxplot(data=dataset_trainCopy,x='Item_Outlet_Sales',y='Item_Weight')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

dataset_trainCopy['Item_Fat_Content'] = le.fit_transform(dataset_trainCopy['Item_Fat_Content'])
dataset_trainCopy['Item_Type'] = le.fit_transform(dataset_trainCopy['Item_Type'])
dataset_testCopy['Item_Fat_Content'] = le.fit_transform(dataset_testCopy['Item_Fat_Content'])
dataset_testCopy['Item_Type'] = le.fit_transform(dataset_testCopy['Item_Type'])

from sklearn.impute import SimpleImputer

impu = SimpleImputer()

imputed_dataset_train = pd.DataFrame(impu.fit_transform(dataset_trainCopy))
imputed_dataset_test = pd.DataFrame(impu.fit_transform(dataset_testCopy))

sb.heatmap(imputed_dataset_train.isnull(),cmap='viridis',yticklabels=False)
sb.heatmap(imputed_dataset_test.isnull(),cmap='viridis',yticklabels=False)
'''
'''
dataset_trainCopyy = dataset_trainCopy.copy()
dataset_trainCopyy.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
'''



dataset_trainCopy = pd.get_dummies(dataset_trainCopy, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet_Identifier'])
dataset_testCopy = pd.get_dummies(dataset_testCopy, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet_Identifier'])


dataset_trainCopy.columns


dataset_trainCopyy = dataset_trainCopy.copy()
dataset_trainCopyy.drop(['Item_Outlet_Sales'],axis=1,inplace=True)


dataset_trainCopyy.dtypes

'''
split dataset into dependent and independent features
'''

X = dataset_trainCopyy.iloc[:,:]
y = dataset_trainCopy.iloc[:,2]


'''
normalization
'''

from sklearn.preprocessing import StandardScaler

SC = StandardScaler()

#dataset_trainCopyy = SC.fit_transform(dataset_trainCopyy)
#dataset_trainCopy = SC.fit_transform(dataset_trainCopy)

X = SC.fit_transform(X)
#y = SC.fit_transform(y)

# normalization dint imprve accuracy

'''
split dataset into train and test
'''

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)


'''
Random Forest Regressor Model implementation 
'''

from sklearn.ensemble import RandomForestRegressor

RFModel = RandomForestRegressor()
RFModel.fit(X_train,y_train)

RFpred = RFModel.predict(X_test)

# RandomForestRegressor - 0.33 accuracy


from sklearn.metrics import r2_score

RFScore = r2_score(y_test,RFpred)
# accuracy of 0.49
# RandomForestRegressor - 0.51 accuracy after normalization

from sklearn import metrics

print("Random Forest Regressor accuracy Score -->",RFScore)

print('MAE for RF:', metrics.mean_absolute_error(y_test, RFpred))
print('MSE for RF:', metrics.mean_squared_error(y_test, RFpred))
print('RMSE for RF:', np.sqrt(metrics.mean_squared_error(y_test, RFpred)))

'''
Linear Regression Model Implementation
'''

from sklearn.linear_model import LinearRegression

LRModel = LinearRegression()

LRModel.fit(X_train,y_train)

LRPred = LRModel.predict(X_test)

LRScore = r2_score(y_test,LRPred)
# accuracy of 0.56

print("Linear Regressor accuracy Score -->",LRScore)

print('MAE for LF:', metrics.mean_absolute_error(y_test, LRPred))
print('MSE for LF:', metrics.mean_squared_error(y_test, LRPred))
print('RMSE for LF:', np.sqrt(metrics.mean_squared_error(y_test, LRPred)))

'''
Xgboost Model Implementation
'''

import xgboost as xgb

XGBModel = xgb.XGBRegressor()
XGBModel.fit(X_train,y_train)

XGBPred = XGBModel.predict(X_test)

XGBScore = r2_score(y_test,XGBPred)
# accuracy of 0.59

print("XGB accuracy Score -->",XGBScore)

print('MAE for XGB:', metrics.mean_absolute_error(y_test, XGBPred))
print('MSE for XGB:', metrics.mean_squared_error(y_test, XGBPred))
print('RMSE for XGB:', np.sqrt(metrics.mean_squared_error(y_test, XGBPred)))
