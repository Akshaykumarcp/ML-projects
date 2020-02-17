# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:32:46 2020

@author: APadashetti
"""

'''
import lib's and dataset
'''

import pandas as pd

df = pd.read_csv('smsspamcollection/SMSSpamCollection',sep='\t',names=['labels','message'])


'''
clean and preprocess
'''
# remove , . to the lower words etc

import re
import nltk
nltk.download('stopwords') # the , if , of ,a ,to to remove

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer # used for stemming i,e technique to find base root of the word
# try lemitization
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

corpus = []

for i in range(0,len(df)): # for all the sentences
    review = re.sub('[^a-zA-Z]', ' ',df['message'][i]) # . , punchuation etc
    review = review.lower() # lower all the words
    review = review.split() # split each sentences where we will get list of sentences
    
    # for stemming. review = [ps.stem(word) for word in review if not word in stopwords.words('english')] # remove's unneccesary words. Get root word (stemming)
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    
    review = ' '.join(review) # join all words into sentences
    corpus.append(review)
    
'''
    
# create BOW model - document matrix
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)  # instead of 6296 , we'll take top most frequent words
X = cv.fit_transform(corpus).toarray() # independent feature
'''

# create TF-IDFmodel

from sklearn.feature_extraction.text import TfidfVectorizer
tfidV = TfidfVectorizer()
X = tfidV.fit_transform(corpus).toarray()


# more columns will be created, after running above code.
# select frequent words  - from 6296 - 
# select random max feature

# select dependent feature

# convert into dummpy variable 

y = pd.get_dummies(df['labels']) # creates two column. dummy variable trap
y = y.iloc[:,1].values # 1 column is enough so remove 1 column

# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# train model using NB classifier

from sklearn.naive_bayes import MultinomialNB

NBModel = MultinomialNB().fit(X_train, y_train)

y_pred=NBModel.predict(X_test)

from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred) # BOW model - 0.98 accuracy with stemming. 0.9820627802690582 accuracy with lemmatization.
# TfidVector Model with lematization gives accuracy of 0.9721973094170404

