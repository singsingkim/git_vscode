import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import time

# data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train,x_test],axis=0)
y = np.concatenate([y_train,y_test],axis=0)
x = x.reshape(x.shape[0],x.shape[1]*x.shape[2])
x = StandardScaler().fit_transform(x)

lda = LinearDiscriminantAnalysis()
x = lda.fit_transform(x,y)
print(x.shape)  # (70000, 9)

evr_sum = np.cumsum(lda.explained_variance_ratio_)
print(evr_sum)
# [0.23718291 0.44064934 0.617996   0.72502353 0.81942663 0.88849806
#  0.93862739 0.97309696 1.        ]

acc_list = []
for i in range(1,x.shape[1]):
    lda = LinearDiscriminantAnalysis(n_components=i)
    lda_x = lda.fit_transform(x,y)

    x_train, x_test, y_train, y_test = train_test_split(lda_x,y,train_size=0.8,random_state=123,stratify=y)

    # model
    model = XGBClassifier()

    # compile fit
    model.fit(x_train,y_train)

    acc = model.score(x_test,y_test)

    print("ACC: ",acc)
    acc_list.append(acc)
    
print(acc_list)
# ACC:  0.9215714335441589
# ACC:  0.9162857142857143
# [0.42228571428571426, 0.5620714285714286, 0.7513571428571428, 0.8301428571428572, 0.8466428571428571, 0.8731428571428571, 0.8945714285714286, 0.9114285714285715]