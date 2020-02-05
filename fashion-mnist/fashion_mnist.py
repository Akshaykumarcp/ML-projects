# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 05:37:54 2020

@author: Akshay kumar C P
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

df_train = pd.read_csv('fashionmnist/fashion-mnist_train.csv')
df_test = pd.read_csv('fashionmnist/fashion-mnist_test.csv')

'''
visualization
'''
# images are flatened into each row

df_train.head()

training = np.array(df_train,dtype = 'float32')
testing = np.array(df_test,dtype = 'float32')

# display images randomly

import random
i = random.randint(1,60000)

plt.imshow(training[i,1:].reshape(28,28))

# display label
label = training[i,0]
label

'''
Each training and test example is assigned to one of the following labels:

0 T-shirt/top
1 Trouser
2 Pullover
3 Dress
4 Coat
5 Sandal
6 Shirt
7 Sneaker
8 Bag
9 Ankle boot
'''

# let's view data in matric format

