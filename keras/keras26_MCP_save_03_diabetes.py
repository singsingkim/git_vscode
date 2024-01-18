# 14_3 카피
from keras.models import Sequential, load_model
from keras.layers import Dense
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


# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

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


print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

# 만들기
# R2 0.62 이상

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

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
filepath = "".join([path,'k26_03_diabetes_', date,'_', filename])
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


print("============ hist =============")
print(hist)
print("===============================")
# print(datasets) #데이터.데이터 -> 데이터.타겟
print(hist.history)     # 오늘과제 : 리스트, 딕셔너리=키(loss) : 똔똔 밸류 한 쌍괄호{}, 튜플
                                    # 두 개 이상은 리스트
                                    # 딕셔너리
print("============ loss =============")
print(hist.history['loss'])
print("============ val_loss =========")
print(hist.history['val_loss'])
print("===============================")

print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2),"초")
print("RMSE : ", rmse)


'''
# ★ 시각화 ★
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

plt.title('당뇨병 로스')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

'''

# 민맥스스케일
# 로스 :  3338.86767578125
# R2 스코어 :  0.523575832341706
# 걸린 시간 :  3.7 초
# RMSE :  57.78293812850216

# 맥스앱스스케일
# 로스 :  3176.3603515625
# R2 스코어 :  0.5467640834146357
# 걸린 시간 :  1.88 초
# RMSE :  56.35920885473039

# 스탠다드스케일
# 로스 :  3458.138916015625
# R2 스코어 :  0.5065570172543277
# 걸린 시간 :  2.02 초
# RMSE :  58.80594286285106

# 로부투스스케일
# 로스 :  3280.69970703125
# R2 스코어 :  0.531875899250148
# 걸린 시간 :  2.43 초
# RMSE :  57.27739124394522