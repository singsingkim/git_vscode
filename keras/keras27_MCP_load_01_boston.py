import warnings


import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time

# 1
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, test_size=0.3,
    shuffle=True, random_state=4)

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


# 2
# model = Sequential()
# model.add(Dense(64, input_dim = 13))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))

# model=load_model('..\Study\_data\_save\MCP\keras26_1_MCP.hdf5')
model=load_model('..\_data\_save\MCP\keras26_1_MCP.hdf5')
model.summary()


# 3
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1, restore_best_weights=True)

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath='..\study\_data\_save\MCP\keras26_1_MCP.hdf5'
# )

# model.compile(loss='mse', optimizer='adam')

# hist = model.fit(x_train, y_train,
#           callbacks=[es, mcp],
#           epochs=10, batch_size=10, validation_split=0.2)
# model.save_weights("..\_data\_save\keras26_1_save_weights.h5")



# 4
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("=======================================")
# print(hist.history['loss'])
# print(hist.history['val_loss'])


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

