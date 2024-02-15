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
    print(x.shape)  # (70000, 9)

    x_train, x_test, y_train, y_test = train_test_split(lda_x,y,train_size=0.8,random_state=123,stratify=y)

    # model
    model = Sequential()
    model.add(Dense(256, input_shape=x_train.shape[1:], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # compile fit
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='acc')
    model.fit(x_train,y_train, epochs=100, batch_size=512, verbose=2)

    loss = model.evaluate(x_test,y_test)

    print("ACC: ",loss[1])
    acc_list.append(loss[1])
    
print(acc_list)
# ACC:  0.9215714335441589
# [0.4263571500778198, 0.5670714378356934, 0.7526428699493408, 0.8363571166992188, 0.8535000085830688, 0.8775714039802551, 0.8972142934799194, 0.914642870426178]