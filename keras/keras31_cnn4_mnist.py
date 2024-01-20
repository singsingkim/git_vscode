# acc = 0.98 이상

import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D,Flatten
from sklearn.preprocessing import OneHotEncoder

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(
    x_train.shape, y_train.shape
)  # (60000, 28, 28) : ==> 흑백 (60000, 28,28, 1)인데 생략 //(60000,)
print(x_test.shape, y_test.shape)  # (10000, 28, 28) //(10000,)\

print(x_train[0])
unique, count = np.unique(y_train, return_counts=True)
print(
    unique, count
)  # [0 1 2 3 4 5 6 7 8 9] [5923 6742 5958 6131 5842 5421 5918 6265 5851 5949]
print(pd.value_counts(y_test))
"""
1    1135
2    1032
7    1028
3    1010
9    1009
4     982
0     980
8     974
6     958
5     892
"""

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = one_hot.transform(y_test.reshape(-1, 1)).toarray()

# print(x_train[0].shape)#(60000, 28, 28, 1)
# print(x_test[0].shape)#(10000, 28, 28, 1)

#2. 모델
model = Sequential()
model.add(Conv2D(9, (2,2), input_shape = (28, 28, 1))) 
model.add(Conv2D(16, (3,3), activation='relu')) #전달 (N,25,25,10)
model.add(Conv2D(32,(4,4))) #전달 (N,22,22,15)
model.add(Flatten()) #평탄화
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
'''
(kernel_size * channels + bias) * filters 아웃풋의 값은 인풋의 채널 수 와 같다.
1번째 레이어  = (4 * 1 + 1) * 9 = 45
2번째 레이어 = (9 * 9 + 1) * 10 = 810
3번째 레이어 = (16 * 10  + 1 ) * 15 = 2415
'''



#컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1000, verbose= 1, epochs= 1000, validation_split=0.2 )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

y_test_armg =  np.argmax(y_test, axis=1)
predict = np.argmax(model.predict(x_test),axis=1)
print(predict)

# loss =  0.3176672160625458
# acc =  0.9800999760627747
# 313/313 [==============================] - 0s 703us/step
# [7 2 1 ... 4 5 6]


'''
model = Sequential()
model.add(Conv2D(9, (2,2), input_shape = (28, 28, 1))) 
model.add(Conv2D(40, (3,3), activation='relu')) #전달 (N,25,25,10)
model.add(Conv2D(15,(4,4))) #전달 (N,22,22,15)
model.add(Flatten()) #평탄화
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1000, verbose= 1, epochs= 400, validation_split=0.2 )


model = Sequential()
model.add(Conv2D(9, (2,2), input_shape = (28, 28, 1))) 
model.add(Conv2D(40, (3,3), activation='relu')) #전달 (N,25,25,10)
model.add(Conv2D(15,(4,4))) #전달 (N,22,22,15)
model.add(Flatten()) #평탄화
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
loss =  0.15593811869621277
acc =  0.9815999865531921


model = Sequential()
model.add(Conv2D(9, (2,2), input_shape = (28, 28, 1))) 
model.add(Conv2D(32, (3,3), activation='relu')) #전달 (N,25,25,10)
model.add(Conv2D(16,(4,4))) #전달 (N,22,22,15)
model.add(Flatten()) #평탄화
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=10000, verbose= 1, epochs= 500, validation_split=0.2 )
loss =  0.18924179673194885
acc =  0.9858999848365784
'''