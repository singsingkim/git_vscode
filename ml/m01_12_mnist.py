import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVC

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(x_train)
print(x_train[0])
print(y_train[0])   # 5
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
print(pd.value_counts(y_test))

# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892
'''
x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# print(x_train.shape[0]) # 60000
'''
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)

###### 스케일링
 
# x_train = x_train.reshape(60000, 28*28)/255.
# x_test = x_test.reshape(10000, 28*28)/255.
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
x_train = x_train/255.0
x_test = x_test/255.0
## 위 두 줄과 네 줄 모두 같다


### 리쉐이프를 위에서 해주어야 실행된다
# x_train = (x_train - 127.5)/127.5
# x_test = (x_test - 127.5)/127.5


# # 리쉐이프를 위에서 해주어야 실행된다
# scaler = MinMaxScaler()

# scaler.fit(x_train)
# x_train=scaler.transform(x_train)
# x_test=scaler.transform(x_test)

# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

# ohe = OneHotEncoder(sparse=False)
# y_train = ohe.fit_transform(y_train)
# y_test = ohe.fit_transform(y_test)

# 2. 모델 순차형
model = LinearSVC(C = 100)
# model = Sequential()
# model.add(Dense(100, input_shape=(784,), activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(60, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(30, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# 모델 함수형
# input1 = Input(shape=(28*28,))
# dense1 = Dense(100, activation='relu')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(10, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(100, activation='relu')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(100, activation='relu')(drop3)
# dense5 = Dense(60, activation= 'relu')(dense4)
# drop4 = Dropout(0.3)(dense5)
# dense6 = Dense(30, activation= 'relu')(drop4)
# output1 = Dense(10, activation= 'softmax')(dense6)
# model = Model(inputs=input1, outputs=output1)



# model.summary()

# print(x_train.shape, y_train.shape)


# 3. 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss= 'categorical_crossentropy', 
#               optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, batch_size=64, 
#           verbose= 1, epochs= 10, validation_split=0.2 )

# 4.평가, 예측
results = model.score(x_test, y_test)
print('results : ', results)
# results = model.evaluate(x_test, y_test)
# print('loss = ', results[0])
# print('acc = ', results[1])

# y_test_armg =  np.argmax(y_test, axis=1)
# predict = np.argmax(model.predict(x_test),axis=1)

# 순차형 모델 에포 10
# loss =  1.72505784034729
# acc =  0.38589999079704285

# 함수형 모델 에포 10
# loss =  1.06037437915802
# acc =  0.5403000116348267

# 함수형 모델 스케일링 /255.
# loss =  0.17791621387004852
# acc =  0.9563999772071838

# 함수형 모델 스케일링 민맥스
# loss =  0.16700047254562378
# acc =  0.9567999839782715
