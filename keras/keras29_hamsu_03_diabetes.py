# 14_3 카피
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error    # 결정계수
import numpy as np
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 1004)


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


print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

# 만들기
# R2 0.62 이상

#2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim = 10))
# model.add(Dropout(0.2))
# model.add(Dense(32))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dropout(0.2))
# model.add(Dense(1))


# 2. 모델구성 함수형
input1 = Input(shape=(10,))
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



#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint       # 클래스는 정의가 필요
import datetime
date = datetime.datetime.now()
print(date)         # 2024-01-17 10:54:10.769322
print(type(date))   # <class 'datetime.datetime')
date = date.strftime("%m%d_%H%M")
print(date)         # 0117_1058
print(type(date))   # <class 'str'>

path='..\_data\_save\MCP\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
filepath = "".join([path,'k28_03_diabetes_', date,'_', filename])
# '..\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
            mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
            patience=100,      # 최소값 찾은 후 열 번 훈련 진행
            verbose=1,
            restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
            )

mcp = ModelCheckpoint(
    monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
    filepath=filepath
    )

model.compile(loss = 'mse', optimizer = 'adam')

hist = model.fit(x_train, y_train, epochs = 1000, 
            batch_size = 10, validation_split=0.3,
            verbose=1, callbacks=[es,mcp]
            )

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)
r2 = r2_score(y_test, y_predict)    # 결정계수

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)


print(hist.history['val_loss'])

print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("RMSE : ", rmse)


# 로스 :  3524.18701171875
# R2 스코어 :  0.4971325721412544
# RMSE :  59.36486465661254
