# https://dacon.io/competitions/open/235576/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    # rmse 사용자정의 하기 위해 불러오는것
import time                

#1. 데이터

path = "c:/_data/dacon/ddarung//"

# print(path + "aaa.csv") # c:/_data/dacon/ddarung/aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
# \ \\ / // 다 가능
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "submission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

print(train_csv.shape)          # (1459, 10)
print(test_csv.shape)           # (715, 9) 아래 서브미션과의 열의 합이 12 인것은 id 열 이 중복되어서이다
print(submission_csv.shape)     # (715, 2)

print(train_csv.columns)
# (['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())    # 평균,최소,최대 등등 표현 # DESCR 보다 많이 활용되는 함수. 함수는 () 붙여주어야 한다 이게 디폴트값

######### 결측치 처리 1. 제거 #########
train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
######### 결측치 처리 2. 0으로 #########
# train_csv = train_csv.fillna(0)   # 결측치 행에 0을 집어 넣는다

# print(train_csv.isnull().sum())
print(train_csv.isna().sum())       # 위 와 같다. isnull() = isna()
print(train_csv.info())
print(train_csv.shape)


test_csv = test_csv.fillna(test_csv.mean())     # 널값에 평균을 넣은거
print(test_csv.info())


######### x 와 y 를 분리 #########
x = train_csv.drop(['count'], axis = 1)     # count를 삭제하는데 count가 열이면 액시스 1, 행이면 0
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=True, 
            train_size= 0.7, random_state= 666
            )
print(x_train.shape, x_test.shape)  # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (929,) (399,)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 9))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()

from keras.callbacks import EarlyStopping       # 클래스는 정의가 필요
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
            mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
            patience=500,      # 최소값 찾은 후 열 번 훈련 진행
            verbose=1
            )

hist = model.fit(x_train, y_train, epochs = 6000,
            batch_size = 200, validation_split=0.2,
            verbose=1, callbacks=[es]
            )
end_time = time.time()
#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
# r2 = r2_score(y_test, y_submit)

print(y_submit)

print(y_submit.shape)   # (715, 1)
print("========================================")
######## submission.csv 만들기(count 컬럼에 값만 넣어주면 됌) ########
submission_csv['count'] = y_submit
print(submission_csv)

submission_csv.to_csv(path + "submission_0109_1.csv", index = False)

y_predict=model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("로스 : ", loss)
print("R2 스코어 : ", r2)

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
print("RMSE : ", rmse)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

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

plt.title('따릉이 로스')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
'''

## train 결측 제거, test 결측 평균 ##
# validation
# [715 rows x 2 columns]
# 13/13 [==============================] - 0s 500us/step
# 로스 :  2884.665771484375
# R2 스코어 :  0.5879407222901258

# 로스 :  2561.645751953125
# R2 스코어 :  0.5949986325343113
# RMSE :  50.61270279985921
# 걸린시간 :  22.7 초
