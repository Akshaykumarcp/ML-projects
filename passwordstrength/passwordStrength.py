# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 06:19:07 2020

@author: Akshay kumar C P
"""

import pandas as pd
import numpy as np

# ',',error_bad_lines=False --> , seperated values are not available will skip those lines
df = pd.read_csv('data.csv',',',error_bad_lines=False)

df.head()

# check null values

df.isnull().sum()

# in password feature there is 1 null values. Let's see which row it is 

df[df['password'].isnull()]

# since there is only 1 null value , Let's drop it

df.dropna(inplace=True)

# dropped null value. Now let's verify whether the row is deleted or not!

df.isnull().sum()

# let's convert df to array for manupulating data

passwords_tuple = np.array(df)

passwords_tuple

# Let's re-shuffle dataset because data is in same order

import random

# shuffling randomly for robustness
random.shuffle(passwords_tuple)

# 1st column is independent and 2nd column is dependent feature

y = [labels[1] for labels in passwords_tuple]

X = [labels[0] for labels in passwords_tuple]

# can also do using pandas without converting into tuple. Above is one of the other approach

'''
Visualization
'''

import seaborn as sb

sb.countplot(x='strength',data=df,palette='RdBu_r')

# Let's convert X (words) into vectors

from sklearn.feature_extraction.text import TfidfVectorizer

# for sentences and words --> vectors will be created 
# here it is words -  need to work on character by character

def word_divide_char(inputs):
    charaters = []
    for i in inputs:
        charaters.append(i)
    return charaters
 
vectorizer = TfidfVectorizer(tokenizer=word_divide_char)

# all the input from passwords -- determince hw many character have and decide how many vectors shud be created

X = vectorizer.fit_transform(X)

X.shape

# 124 features , vector for each character as a seperate feature

# for all 124 features, all the vocabulory
vectorizer.vocabulary_

df.iloc[0,0]

# let's c how above password is converted into features

feature_names = vectorizer.get_feature_names()

# get tfidf vector for first document
first_document_vector=X[0]

# print the scores

tfidfVec = pd.DataFrame(first_document_vector.T.todense(),index=feature_names,columns=['tfidf'])

passwordAsFeatues = tfidfVec.sort_values(by=['tfidf'],ascending=False)

passwordAsFeatues.columns

passwordAsFeatues.shape

# Remember! X is vectors of values

# Now! dependent and independent features are ready. Let's implement model

## logistic refression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=40)

# logistic regression is for binary classifier, with specific parameter change we can convert into multi-class classifier

## for multi_class='ovr'

LRModel = LogisticRegression(penalty='l2',multi_class='ovr')
LRModel.fit(X_train,y_train)

print(LRModel.score(X_test,y_test))

## multi_class='multinomial'

LRModel2 = LogisticRegression(random_state=0, multi_class='multinomial',solver='newton-cg')
LRModel2.fit(X_train,y_train)

print(LRModel2.score(X_test,y_test))

# Let's input password to predict

X_predict = np.array(["akshay123@"])
X_predict=vectorizer.transform(X_predict)
y_pred = LRModel.predict(X_predict)
print(y_pred)
