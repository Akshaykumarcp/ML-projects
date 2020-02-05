# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 04:37:47 2020

@author: Akshay kumar C P
"""

# download dataset from https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

'''
take aways - data science life cyle i,e 
 
 1. data analysis
 2. feature engineering
 3. feature selection
 4. model building
 5. model deployment
'''

# Let's begin with 1. data analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df_train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

df_train.shape

'''
what we do in data analysis  !!!!

1. missing values
2. all numerical variables
3. distribution of numerical variables
4. categoricalvariables
5. cardinality of categorical variables
6. outliers
7. relationship between independent and dependent features 

'''

# missing values present

df_train.isnull().sum()

# find % of nan values present in each features

# list of featues which has missing values

features_with_na = [features for features in df_train.columns if df_train[features].isnull().sum() > 1]

# print feature name and % of missing values

for feature in features_with_na:
    print(feature, np.round(df_train[feature].isnull().mean(),4),'% missing values')
    
# since there is missing values in dataset, let's c any relationship is present related to price feature (traget variable)
    
for feature in features_with_na:
    data = df_train.copy()
    
    # let's make variavle that indicates 1 if observation was missing or zero otherwise
    # for count plot
    data[feature] = np.where(data[feature].isnull(),1,0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
    
'''
Here With the relation between the missing values and the dependent variable is clearly visible.So We need to replace these nan values with something meaningful which we will do in the Feature Engineering section
'''

# ID feature is not required

print('ID of houses {}'.format(len(df_train.Id)))

# numerical variables
# != object
numerical_features = [feature for feature in df_train.columns if df_train[feature].dtypes != 'O']
    
print('Number of numerical variables: ', len(numerical_features))

# visualise the numerical variables
df_train[numerical_features].head()

'''
observation : Temporal Variables(Eg: Datetime Variables)
From the Dataset we have 4 year variables. We have extract information from the datetime variables like no of years or no of days. One example in this specific scenario can be difference in years between the year the house was built and the year the house was sold. We will be performing this analysis in the Feature Engineering which is the next video.
'''

# find out year based features from dataset

year_features = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

# let's print the years

for feature in year_features:
    print(feature, df_train[feature].unique())
    
# let's analyze the temporal datetime variable
## We will check whether there is a relation between year the house is sold and the sales price

df_train.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title("House Price vs YearSold")


## Here we will compare the difference between All years feature with SalePrice
# based on age of building, density is distributed
for feature in year_features:
    if feature!='YrSold':
        data=df_train.copy()
        ## We will capture the difference between year variable and year the house was sold for
        data[feature]=data['YrSold']-data[feature]

        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()

## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(df_train[feature].unique())<25 and feature not in year_features+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))

df_train[discrete_feature].head()

## Lets Find the realtionship between discrete features and Sale PRice

for feature in discrete_feature:
    data=df_train.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()

continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_features+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


## Lets analyse the continuous values by creating histograms to understand the distribution

for feature in continuous_feature:
    data=df_train.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()
    
'''
for continuos features are there skewed data and normally distributed data. So let's perform log normalisation i,e convert from non - gaussian distribution to gausian distribution
'''

## We will be using logarithmic transformation


for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]=np.log(data[feature])
        data['SalePrice']=np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()
        
# check outliers in continous features
# work only for numerical features

for feature in continuous_feature:
    data=df_train.copy()
    if 0 in data[feature].unique():
        pass        
    else:
        data[feature]=np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        
# categorical features
        
categorical_features = [feature for feature in df_train.columns if df_train[feature].dtypes=='O']

df_train[categorical_features].head()

# since there are many categorical features, first thing is find cardinality values i,e how many different categories available in each features

for feature in categorical_features:
    print('The feature is {}, number of categories are {} and categories are {}'.format(feature,len(df_train[feature].unique()),df_train[feature].unique()))
    

# Find out the relationship between categorical variable and dependent feature
    
    for feature in categorical_features:
        data=df_train.copy()
        data.groupby(feature)['SalePrice'].median().plot.bar()
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()