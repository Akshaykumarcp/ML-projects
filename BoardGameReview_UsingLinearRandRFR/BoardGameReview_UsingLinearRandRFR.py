# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 10:57:36 2020

@author: Akshay kumar C P
"""
'''
Reviews can make or break a product; as a result, many companies take drastic 
measures to ensure that their product receives good reviews. When it comes to board games,
 reviews and word-of-mouth are everything. In this project, we will be using a linear regression 
 model to predict the average review a board game will receive based on characteristics such as 
 minimum and maximum number of players, playing time, complexity, etc.
'''


'''
Models

1. linear regression (linear nodel)
2. random forest regressor (non-linear model)

'''

# version used

import sys
import pandas
import matplotlib
import seaborn
import sklearn

print(sys.version)
print("panda version : ",pandas.__version__)
print("matplotlib version : ",matplotlib.__version__)
print("seaborn version : ",seaborn.__version__)
print("sklearn version : ",sklearn.__version__)

# import lib's

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# load data

games = pandas.read_csv("games.csv")

games.columns

games.shape

# histogram of all the ratings in the average_rating column

plt.hist(games["average_rating"])

# print the first row of all the games with zero scores
games[games['average_rating'] == 0].iloc[0]

# print the first row of all the games withscores greather than 0
games[games['average_rating'] > 0 ].iloc[0]


# games[games["average_rating"] == 0]

# Print the first row of all the games with zero scores.
# The .iloc method on dataframes allows us to index by position.
print(games[games["average_rating"] == 0].iloc[0])
# Print the first row of all the games with scores greater than 0.
print(games[games["average_rating"] > 0].iloc[0])

# Remove any rows without user reviews.
games = games[games["users_rated"] > 0]
# Remove any rows with missing values.
games = games.dropna(axis=0)

# Make a histogram of all the ratings in the average_rating column.
plt.hist(games["average_rating"])

# Show the plot.
plt.show()

#correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

# Get all the columns from the dataframe.
columns = games.columns.tolist()
# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on.
target = "average_rating"

# Import a convenience function to split the sets.
from sklearn.model_selection import train_test_split

# Generate the training set.  Set random_state to be able to replicate results.
train = games.sample(frac=0.8, random_state=1)
# Select anything not in the training set and put it in the testing set.
test = games.loc[~games.index.isin(train.index)]
# Print the shapes of both sets.
print(train.shape)
print(test.shape)

# Import the linear regression model.
from sklearn.linear_model import LinearRegression

# Initialize the model class.
model = LinearRegression()
# Fit the model to the training data.
model.fit(train[columns], train[target])

# Import the scikit-learn function to compute error.
from sklearn.metrics import mean_squared_error

# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
mean_squared_error(predictions, test[target])

# 2.0787767119035703

# Import the random forest model.
from sklearn.ensemble import RandomForestRegressor

# Initialize the model with some parameters.
model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
# Fit the model to the data.
model.fit(train[columns], train[target])
# Make predictions.
predictions = model.predict(test[columns])
# Compute the error.
mean_squared_error(predictions, test[target])

# 1.4458560046071653


