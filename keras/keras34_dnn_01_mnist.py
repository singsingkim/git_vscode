import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.preprocessing import OneHotEncoder

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


print(x_train.shape, x_test.shape)  # (60000, 784) (10000, 784)

y_train=y_train.reshape(-1,1)
y_test=y_test.reshape(-1,1)

ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train)
y_test = ohe.fit_transform(y_test)

# 2. 모델
model = Sequential()
model.add(Dense(100, input_shape=(784,), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

print(x_train.shape, y_train.shape)

# 3. 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=64, 
          verbose= 1, epochs= 4000, validation_split=0.2 )

# 4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

# y_test_armg =  np.argmax(y_test, axis=1)
# predict = np.argmax(model.predict(x_test),axis=1)