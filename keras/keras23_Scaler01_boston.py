import warnings


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time

# 데이터
datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape)  # (506, 13) 506 행, 13 열
print(y)
print(y.shape)  # (506, ) 506 스칼라

print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

print(datasets.DESCR)

# [실습]
# train_size 0.7 이상, 0.9 이하
# R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=4)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 0.0
# print(np.min(x_test))   # -0.010370370370370367
# print(np.max(x_train))  # 1.0
# print(np.max(x_test))   # 1.0789745710150918

# scaler = MaxAbsScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 



model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=10)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# 민맥스스케일
# 로스 :  30.428585052490234
# R2 스코어 :  0.7085944883148481
# 걸린시간 :  10.57 초

# 맥스앱스
# 로스 :  29.773719787597656
# R2 스코어 :  0.7148659339617173
# 걸린시간 :  10.19 초

# 스탠다드스케일
# 로스 :  32.773616790771484
# R2 스코어 :  0.6861368219450654
# 걸린시간 :  10.33 초

# 로부스트스케일
# 로스 :  31.504825592041016
# R2 스코어 :  0.6982876624705969
# 걸린시간 :  10.34 초