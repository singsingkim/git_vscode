# 14_2 카피
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)   # 20640 행 , 8 열

# [실습] 만들기
# R2 0.55 ~ 0.6 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
    train_size = 0.7,
    test_size = 0.3,
    shuffle = True,
    random_state = 1)

model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()

from keras.callbacks import EarlyStopping       # 클래스는 정의가 필요
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
    mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
    patience=10,      # 최소값 찾은 후 열 번 훈련 진행
    verbose=1
    )

hist = model.fit(x_train, y_train, epochs = 10, 
    batch_size=50, validation_split=0.3, 
    verbose=1, callbacks=[es]
    )

end_time = time.time()

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa,bbb))
rmse = RMSE(y_test, y_predict)

print("RMSE : ", rmse)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

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
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# ★ 시각화 ★
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
import matplotlib.font_manager as fm
font_path = "c:\Windows\Fonts\MALGUN.TTF"
font_name=fm.FontProperties(fname=font_path).get_name()
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.title('캘리포니아 로스')        # 한글깨짐 해결할것
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

# epochs = 2000 , batch_size = 50 , random_state = 1

# Epoch 2000/2000
# 289/289 [==============================] - 0s 521us/step - loss: 0.5747
# 194/194 [==============================] - 0s 438us/step - loss: 0.5791
# 194/194 [==============================] - 0s 345us/step
# 645/645 [==============================] - 0s 389us/step
# 로스 :  0.5790699124336243
# R2 스코어 :  0.5595365853230052
# 걸린시간 :  306.56 초

# validation
# 194/194 [==============================] - 0s 418us/step - loss: 0.5895
# 194/194 [==============================] - 0s 407us/step
# 645/645 [==============================] - 0s 395us/step
# 로스 :  0.5895447134971619
# R2 스코어 :  0.5515689966644968
# 걸린시간 :  291.85 초