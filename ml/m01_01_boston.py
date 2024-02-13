# 드롭 아웃 
import warnings
import numpy as np
import pandas as pd
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Input, Conv2D, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.svm import LinearSVR
# 1
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, 
    shuffle=True, random_state=4567)

print(x.shape, y.shape) # (506, 13) (506,)

print(x_train.shape, y_train.shape) # (1, 13) (1,)
print(x_test.shape, y_test.shape) # (505, 13) (505,)
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1, 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1, 1)
print(x_train.shape)    # (354, 13, 1, 1)
print(x_test.shape)     # (152, 13, 1, 1)

print(x_train[0])
print(y_train[0])
# unique, count = np.unique(y_train, return_counts=True)
print(y_train)
# print(unique, count)


print(np.unique(x_train, return_counts=True))
print(y_train.shape, y_test.shape)
# print(pd.value_counts(x_train))
# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train.reshape(-1,1)).toarray()
# y_test = ohe.fit_transform(y_test.reshape(-1,1)).toarray()
print(x_train.shape, y_train.shape) # (354, 13, 1, 1) (354,)

# 2 모델
model = LinearSVR(C = 100)
model.fit(x_train, y_train)
# model = Sequential()
# model.add(Conv1D(256, 2, input_shape = (13, 1))) 
# # model.add(MaxPooling2D())
# model.add(Flatten()) #평탄화
# model.add(Dense(16))
# model.add(Dense(128))
# model.add(Dense(32))
# # model.add(MaxPooling2D())
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# model.summary()

# model.summary()



# # # 3
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# print(date)         # 2024-01-17 10:54:10.769322
# print(type(date))   # <class 'datetime.datetime')
# date = date.strftime("%m%d_%H%M")
# print(date)         # 0117_1058
# print(type(date))   # <class 'str'>

# path='c:\_data\_save\MCP\\'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
# filepath = "".join([path,'k54_Conv1D_01_boston', date,'_', filename])

# es = EarlyStopping(monitor='val_loss', mode='min',
#                    patience=100, verbose=0, restore_best_weights=True)

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath=filepath
# )

# model.compile(loss='mse', optimizer='adam')    
# start_time = time.time()
# hist = model.fit(x_train, y_train,
#           callbacks=[es, mcp],
#           epochs=10, batch_size=32, validation_split=0.2)
# end_time = time.time()
# 4
print("==================== 1. 기본 출력 ======================")

# results = model.evaluate(x_test, y_test, verbose=1)
results = model.score(x_test, y_test)
print("로스 : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("R2 스코어 : ", r2)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")
# print(y_predict)
# print(y_test_armg)

# 로스 :  22.258581161499023
# 5/5 [==============================] - 0s 1ms/step
# R2 스코어 :  0.7366618376879629
# 걸린 시간 :  17.68 초