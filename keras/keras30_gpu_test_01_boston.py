# 드롭 아웃 
import warnings
import numpy as np
from keras.models import Sequential, Model,load_model
from keras.layers import Dense, Dropout, Input
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
    shuffle=True, random_state=4567)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 


# # 2 모델구성 순차형
# model = Sequential()
# model.add(Dense(64, input_dim = 13))
# model.add(Dropout(0.2))    # 디폴트값 // 13 이 들어가서 64개가 나온후에 0.2 퍼센트를 없앤다.
# model.add(Dense(32))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))    # 디폴트값 // 16개가 나온후에 0.3 퍼센트를 없앤다.
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dropout(0.4))    # 디폴트값 // 4개가 나온후에 0.5 퍼센트를 없앤다.
# model.add(Dense(1))

# 2. 모델구성 함수형
input1 = Input(shape=(13,))
dense1 = Dense(64)(input1)
drop1 = Dropout(0.2)(dense1)
dense2 = Dense(32)(dense1)
dense3 = Dense(16, activation='relu')(drop1)
drop2 = Dropout(0.3)(dense3)
dense4 = Dense(8)(drop2)
dense5 = Dense(4)(dense4)
drop3 = Dropout(0.2)(dense5)
output1 = Dense(1)(drop3)
model = Model(inputs=input1, outputs=output1)




# model.summary()


# # 3
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date)         # 2024-01-17 10:54:10.769322
print(type(date))   # <class 'datetime.datetime')
date = date.strftime("%m%d_%H%M")
print(date)         # 0117_1058
print(type(date))   # <class 'str'>

path='..\_data\_save\MCP\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
filepath = "".join([path,'k30_01_boston', date,'_', filename])

es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=0, restore_best_weights=True)

mcp = ModelCheckpoint(
    monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
    filepath=filepath
)

model.compile(loss='mse', optimizer='adam')

start_time = time.time()
hist = model.fit(x_train, y_train,
          callbacks=[es, mcp],
          epochs=10000, batch_size=10, validation_split=0.2)
end_time = time.time()
# 4
print("==================== 1. 기본 출력 ======================")

loss = model.evaluate(x_test, y_test, verbose=0)
print("로스 : ", loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_predict, y_test)
print("R2 스코어 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2),"초")


# 로스 :  39.29941940307617
# 5/5 [==============================] - 0s 0s/step
# R2 스코어 :  0.2886246577359758
