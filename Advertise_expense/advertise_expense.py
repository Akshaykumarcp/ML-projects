# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:34:59 2020

@author: APadashetti
"""
# Linear Regression

'''
The Problem statement:

This data is about the amount spent on advertising through different channels like TV, Radio and Newspaper. 
The goal is to predict how the expense on each channel affects the sales and is there a way to optimise that sale?

'''

'''
What are the features?

TV: Advertising dollars spent on TV for a single product in a given market (in thousands of dollars)
Radio: Advertising dollars spent on Radio
Newspaper: Advertising dollars spent on Newspaper
What is the response?

Sales: sales of a single product in a given market (in thousands of widgets)

'''

''' simple linear regression''' 

# necessary Imports
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data= pd.read_csv('Advertising.csv') # Reading the data file

data.head() # checking the first five rows from the dataset

data.shape

data.info() # printing the summary of the dataframe

data.isna().sum() # finding the count of missing values from different columns

# Now, let's showcase the relationship between the feature and target column

# visualize the relationship between the features and the response using scatterplots
fig, axs = plt.subplots(1, 3, sharey=True)
data.plot(kind='scatter', x='TV', y='sales', ax=axs[0], figsize=(16, 8))
data.plot(kind='scatter', x='radio', y='sales', ax=axs[1])
data.plot(kind='scatter', x='newspaper', y='sales', ax=axs[2])


# create X and y
feature_cols = ['TV']
X = data[feature_cols]
y = data.sales

# follow the usual sklearn pattern: import, instantiate, fit
from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=False)
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)

type(y)

lm.predict([[18779789]])

'''

Interpreting the model
How do we interpret the coefficient for spends on TV ad ( 𝛽1 )?

A "unit" increase in spends on a TV ad is associated with a 0.047537 "unit" increase in Sales.
Or, an additional $1,000 on TV ads is translated to an increase in sales by 47.53 Dollars.
As an increase in TV ad expenditure is associated with a decrease in sales,  𝛽1  would be negative.

'''

#calculate the prediction
7.032594 + 0.047537*50

#Thus, we would predict Sales of 9,409 widgets in that market.

#Let's do the same thing using code.

#  Let's create a DataFrame since the model expects it
X_new = pd.DataFrame({'TV': [50]})
X_new.head()

# use the model to make predictions on a new value
lm.predict(X_new)

# Plotting the Least Squares Line

# create a DataFrame with the minimum and maximum values of TV
X_new = pd.DataFrame({'TV': [data.TV.min(), data.TV.max()]})
X_new.head()

# make predictions for those x values and store them
preds = lm.predict(X_new)
preds

# first, plot the observed data
data.plot(kind='scatter', x='TV', y='sales')

# then, plot the least squares line
plt.plot(X_new, preds, c='red', linewidth=2)

'''
Model Confidence
Question: Is linear regression a low bias/high variance model or a high bias/low variance model?

Answer: It's a High bias/low variance model. Even after repeated sampling, the best fit line will stay roughly in the same position (low variance), but the average of the models created after repeated sampling won't do a great job in capturing the perfect relationship (high bias). Low variance is helpful when we don't have less training data!

If the model has calculated a 95% confidence for our model coefficients, it can be interpreted as follows: If the population from which this sample is drawn, is sampled 100 times, then approximately 95 (out of 100) of those confidence intervals shall contain the "true" coefficients.

'''

import statsmodels.formula.api as smf
lm = smf.ols(formula='sales ~ TV', data=data).fit()
lm.conf_int()

# print the p-values for the model coefficients
lm.pvalues

'''
If the 95% confidence interval includes zero, the p-value for that coefficient will be greater than 0.05. If the 95% confidence interval does not include zero, the p-value will be less than 0.05.

Thus, a p-value of less than 0.05 is a way to decide whether there is any relationship between the feature in consideration and the response or not. Using 0.05 as the cutoff is just a convention.

In this case, the p-value for TV ads is way less than 0.05, and so we believe that there is a relationship between TV advertisements and Sales.

Note that we generally ignore the p-value for the intercept.
'''

# How Well Does the Model Fit the data

# print the R-squared value for the model
lm.rsquared

'''
Is it a "good" R-squared value? Now, that’s hard to say. In reality, the domain to which the data belongs to plays a significant role in deciding the threshold for the R-squared value. Therefore, it's a tool for comparing different models.
'''






''' multiple linear regression'''

# create X and y
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data.sales

lm = LinearRegression()
lm.fit(X, y)

# print intercept and coefficients
print(lm.intercept_)
print(lm.coef_)


'''
How do we interpret these coefficients? If we look at the coefficients, the coefficient for the newspaper spends is negative. It means that the money spent for newspaper advertisements is not contributing in a positive way to the sales.

A lot of the information we have been reviewing piece-by-piece is available in the model summary output:
'''

lm = smf.ols(formula='sales ~ TV + radio + newspaper', data=data).fit()
lm.conf_int()
lm.summary()

'''
What are the things to be learnt from this summary?

TV and Radio have positive p-values, whereas Newspaper has a negative one. Hence, we can reject the null hypothesis for TV and Radio that there is no relation between those features and Sales, but we fail to reject the null hypothesis for Newspaper that there is no relationship between newspaper spends and sales.
The expenses on bot TV and Radio ads arepositively associated with Sales, whereas the expense on newspaper ad is slightly negatively associated with the Sales.
This model has a higher value of R-squared (0.897) than the previous model, which means that this model explains more variance and provides a better fit to the data than a model that only includes the TV.
'''

'''
Feature Selection
How do I decide which features have to be included in a linear model? Here's one idea:

Try different models, and only keep predictors in the model if they have small p-values.
Check if the R-squared value goes up when you add new predictors to the model.
What are the drawbacks in this approach? -If the underlying assumptions for creating a Linear model(the features being independent) are violated(which usually is the case),p-values and R-squared values are less reliable.

Using a p-value cutoff of 0.05 means that adding 100 predictors to a model that are pure noise, still 5 of them (on average) will be counted as significant.
R-squared is susceptible to model overfitting, and thus there is no guarantee that a model with a high R-squared value will generalise. Following is an example:
'''
# only include TV and Radio in the model
lm = smf.ols(formula='sales ~ TV + radio', data=data).fit()
lm.rsquared

'''
Selecting the model with the highest value of R-squared is not a correct approach as the value of R-squared shall always increase whenever a new feature is taken for consideration even if the feature is unrelated to the response.

The alternative is to use adjusted R-squared which penalises the model complexity (to control overfitting), but this again generally under-penalizes complexity.

a better approach to feature selection isCross-validation. It provides a more reliable way to choose which of the created models will best generalise as it better estimates of out-of-sample error. An advantage is that the cross-validation method can be applied to any machine learning model and the scikit-learn package provides extensive functionality for that.

Handling Categorical Predictors with Two Categories
Till now, all the predictors have been numeric. What if one of the predictors is categorical?

We’ll create a new feature called Scale, and shall randomly assign observations as small or large:
'''

import numpy as np

# set a seed for reproducibility
np.random.seed(12345)

# create a Series of booleans in which roughly half are True
nums = np.random.rand(len(data))
mask_large = nums > 0.5

# initially set Size to small, then change roughly half to be large
data['Scale'] = 'small'
data.loc[mask_large, 'Scale'] = 'large'
data.head()

'''

For the scikit-learn library, all data must be represented numerically. If the feature only has two categories, we can simply create a dummy variable that represents the categories as a combination of binary value:
'''

# create a new Series called IsLarge
data['IsLarge'] = data.Scale.map({'small':0, 'large':1})
data.head()

# Let's redo the multiple linear regression problem and include the IsLarge predictor:


    # create X and y
feature_cols = ['TV', 'radio', 'newspaper', 'IsLarge']
X = data[feature_cols]
y = data.sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
i=0
for col in feature_cols:
    print('The Coefficient of ',col, ' is: ',lm.coef_[i])
    i=i+1

'''
How do we interpret the coefficient for IsLarge? For a given TV/Radio/Newspaper ad expenditure if the average sales increases by 57.42 widgets, it’s considered as a large market.

What if the 0/1encoding is reversed? Still, the value of the coefficient shall be same, the only difference being the sign. It’ll be a negative number instead of positive.

Handling Categorical variables with More than Two Categories
Let's create a new column called Targeted Geography, and randomly assign observations to be rural, suburban, or urban:

'''
    
# set a seed for reproducibility
np.random.seed(123456)

# assign roughly one third of observations to each group
nums = np.random.rand(len(data))
mask_suburban = (nums > 0.33) & (nums < 0.66)
mask_urban = nums > 0.66
data['Targeted Geography'] = 'rural'
data.loc[mask_suburban, 'Targeted Geography'] = 'suburban'
data.loc[mask_urban, 'Targeted Geography'] = 'urban'
data.head()

# We need to represent the ‘Targeted Geography’ column numerically. But mapping urban=0, suburban=1 and rural=2 will mean that rural is two times suburban which is not the case. Hence, we’ll create another dummy variable:

# create three dummy variables using get_dummies, then exclude the first dummy column
area_dummies = pd.get_dummies(data['Targeted Geography'], prefix='Targeted Geography').iloc[:, 1:]

# concatenate the dummy variable columns onto the original DataFrame (axis=0 means rows, axis=1 means columns)
data = pd.concat([data, area_dummies], axis=1)
data.head()

'''
What does the encoding say?

rural is encoded as Targeted Geography_suburban=0 and Targeted Geography_urban=0
suburban is encoded as Targeted Geography_suburban=1 and Targeted Geography_urban=0
urban is encoded as Targeted Geography_suburban=0 and Targeted Geography_urban=1
Now the question is: Why have we used two dummy columns instead of three?

Because using only two dummy columns, we can capture the information of all the 3 columns. For example, if the value for Targeted Geography_urban as well as Targeted Geography_rural is 0, it automatically means that the data belongs to Targeted Geography_suburban.

This is called handling the dummy variable trap. If there are N dummy variable columns, then the same information can be conveyed by N-1 columns. Let's include the two new dummy variables in the model:
'''

# create X and y
feature_cols = ['TV', 'radio', 'newspaper', 'IsLarge', 'Targeted Geography_suburban', 'Targeted Geography_urban']
X = data[feature_cols]
y = data.sales

# instantiate, fit
lm = LinearRegression()
lm.fit(X, y)

# print coefficients
print(feature_cols, lm.coef_)

'''
How do we interpret the coefficients?

If all other columns are constant, the suburban geography is associated with an average decrease of 106.56 widgets in sales for $1000 spent.

if $1000 is spent in an urban geography, it amounts to an average increase in Sales of 268.13 widgets

A final note about dummy encoding: If we have categories that can be ranked (i.e., worst, bad, good, better, best), we can potentially represent them numerically as (1, 2, 3, 4, 5) using a single dummy column

'''

