# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 09:48:29 2019

@author: APadashetti
"""

#step 1 - import lib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# step 2 --> import dataset

dataset = pd.read_csv('Iris.csv') 

# step 3 --> Analysis

print(dataset.head())



# let's c how many categories of flowers are there / examples are ther for each species

print(dataset['Species'].value_counts())

# Visualization

# let's use scatter ploy for visualization. Pandas provide .plot for plotting

dataset.plot(kind="scatter",x="SepalLengthCm",y="SepalWidthCm")

dataset.plot(kind="scatter",x="PetalLengthCm",y="PetalWidthCm")

# seaborn is used to plot similar kind of plots. seaborn provides additional capabilities
# seaborn jointplot shows bi-variant scatterplot and uni-variant histogram together

sb.jointplot(x="SepalLengthCm",y="SepalWidthCm",data=dataset,size=5)

# additional information about the diagram is missing. so let's add 
# using FacetGrid - color the species by color

sb.FacetGrid(dataset,hue="Species",size=5).map(plt.scatter,"SepalLengthCm","SepalWidthCm").add_legend()

# to look individual feature using seaborn through boxplot

bp=sb.boxplot(x="Species",y="PetalLengthCm",data=dataset)

# One way we can extend this plot is adding a layer of individual points on top of
# it through Seaborn's striplot
# 
# We'll use jitter=True so that all the points don't fall in single vertical lines
# above the species
#
# Saving the resulting axes as ax each time causes the resulting plot to be shown
# on top of the previous axes
# run below line together

bp=sb.boxplot(x="Species",y="PetalLengthCm",data=dataset)
bp=sb.stripplot(x="Species",y="PetalLengthCm",data=dataset,jitter=True,edgecolor="gray")


# A violin plot combines the benefits of the previous two plots and simplifies them
# Denser regions of the data are fatter, and sparser thiner in a violin plot
# violin plot have bloxplot inside it and density is shown via colors and size. Middle dot in the violin is median.

sb.violinplot(x="Species",y="PetalLengthCm",data=dataset,size=6)

# A final seaborn plot useful for looking at univariate relations is the kdeplot,
# which creates and visualizes a kernel density estimate of the underlying feature

sb.FacetGrid(dataset,hue="Species",size=6).map(sb.kdeplot,"PetalLengthCm").add_legend()
sb.FacetGrid(dataset,hue="Species",size=6).map(sb.kdeplot,"PetalWidthCm").add_legend()
sb.FacetGrid(dataset,hue="Species",size=6).map(sb.kdeplot,"SepalLengthCm").add_legend()
sb.FacetGrid(dataset,hue="Species",size=6).map(sb.kdeplot,"SepalWidthCm").add_legend()

#Plot two shaded bivariate densities:

setosa = dataset.loc[dataset.Species == "Iris-setosa"]
virginica = dataset.loc[dataset.Species == "Iris-virginica"]
ax = sb.kdeplot(setosa.SepalWidthCm, setosa.SepalLengthCm,cmap="Reds", shade=True, shade_lowest=False)
ax = sb.kdeplot(virginica.SepalWidthCm, virginica.SepalLengthCm,cmap="Blues", shade=True, shade_lowest=False)

# Another useful seaborn plot is the pairplot, which shows the bivariate relation
# between each pair of features
# 
# From the pairplot, we'll see that the Iris-setosa species is separataed from the other
# two across all feature combinations
sb.pairplot(dataset.drop("Id", axis=1), hue="Species", size=3, diag_kind="kde")

# Now that we've covered seaborn, let's go back to some of the ones we can make with Pandas
# We can quickly make a boxplot with Pandas on each feature split out by species
dataset.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))

# A final multivariate visualization technique pandas has is radviz
# Which puts each feature as a point on a 2D plane, and then simulates
# having each sample attached to those points through a spring weighted
# by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(dataset.drop("Id", axis=1), "Species")

dataset.info()  #checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed

# convert Species column from catecorical to numerical values

from sklearn.preprocessing import LabelEncoder
numbers = LabelEncoder()
dataset['Species_encoded'] = numbers.fit_transform(dataset.Species)
y_train

# split data

X = dataset.iloc[:,:-2]
y = dataset.iloc[:,6]

# splitting dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# MLR algorithm

# fitting MLR to training test
from sklearn.linear_model import LinearRegression

regression = LinearRegression()
regression.fit(X_train,y_train)

# predict the test set

y_pred = regression.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_pred)

'''
dint work - ValueError: Classification metrics can't handle a mix of continuous and multiclass targets
from sklearn import metrics
print('The accuracy of the MLR is:',metrics.accuracy_score(y_pred,y_test))
'''

# model MLR is good, 0.95 accuracy

# SVM algorithm

from sklearn import svm

svmModel = svm.SVC()
svmModel.fit(X_train,y_train)

svmPredict = svmModel.predict(X_test)

from sklearn.metrics import r2_score
score2 = r2_score(y_test,svmPredict)

# model SVM is exellent, 1.0 is the accuracy

print('The accuracy of the SVM is:',metrics.accuracy_score(svmPredict,y_test))


