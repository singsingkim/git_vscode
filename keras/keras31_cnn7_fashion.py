# 0.99 이상 맹글기
from keras.datasets import fashion_mnist
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from sklearn.preprocessing import OneHotEncoder

# 1 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(y_test.shape, y_test.shape)   # (10000,) (10000,)

print(x_train)
print(x_train[0])   # 장화?
print(x_train[1])   # 모르겟음
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
print(pd.value_counts(y_test))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
one_hot = OneHotEncoder()
y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
y_test = one_hot.transform(y_test.reshape(-1, 1)).toarray()

#2. 모델
model = Sequential()
model.add(Conv2D(10, (2,2), input_shape = (28, 28, 1))) 
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
# import matplotlib.pyplot as plt
# plt.imshow(x_train[1], 'gray')
# plt.show()

#3 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, batch_size=1000, verbose= 1, epochs= 1000, validation_split=0.2 )

#4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

y_test_armg =  np.argmax(y_test, axis=1)
predict = np.argmax(model.predict(x_test),axis=1)
print(predict)

# loss =  2.8183517456054688
# acc =  0.8798999786376953