# https://dacon.io/competitions/open/235576/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error    # rmse 사용자정의 하기 위해 불러오는것
import time                

#1. 데이터

path = "c:/_data/kaggle/bike//"

# print(path + "aaa.csv") # c:/_data/dacon/ddarung/aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
# \ \\ / // 다 가능
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sampleSubmission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

print(train_csv.shape)          # (10886, 11)
print(test_csv.shape)           # (6493, 8)
print(submission_csv.shape)     # (6493, 2)
print(train_csv.columns)        # 열의 이름

# 
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


# test_csv = test_csv.fillna(test_csv.mean())       # 널값에 평균을 넣은거
# test_csv = test_csv.fillna(test_csv.dropna())     # 널값 행을 삭제한거
test_csv = test_csv.fillna(0)                       # 널값에 0 을 넣은거
print(test_csv.info())


######### x 와 y 를 분리 #########
x = train_csv.drop(['count','casual','registered'], axis = 1)     # count를 삭제하는데 count가 열이면 액시스 1, 행이면 0
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
                                                    x, y, shuffle=True, train_size= 0.7, random_state= 4444
)
print(x_train.shape, x_test.shape)  # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)  # (7620,) (3266,)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 8, activation='relu'))  # 다음 레이어에 던지기 전에 활성화하는 함수
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))     # y = wx + b 결과가 음수가 나올때 다음 레이어에 음수를 주지 않기 위해 Relu 활성화 함수를 사용
                        # 모든 연산이 프레딧 된 지점 # 디폴트로 리니어 설정중
                        # 기온이라 예를 든다면 렐루 햇을때 마이너스값이 없이 겨울에 영하 온도가 없게 된다
                        # 기본적으로 최종 연산에는 렐루를 반드시는 아니지만 기본적으로 사용하지 않는다
                        # 최종프레딧에는 보통 softmax 사용

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 40,validation_split= 0.3, verbose=1)
end_time = time.time()
#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
y_submit = model.predict(test_csv)
# y_submit[y_submit < 0] = 0      # 결과가 나온값에 후처리를 한 것[프레딧 한 값이 그대로 나와야 하는데 이렇게하면 좋지 않다]

print(y_submit)
print(y_submit.shape)   # (6493, 1)
print("========================================")
######## submission.csv 만들기(count 컬럼에 값만 넣어주면 됌) ########
submission_csv['count'] = y_submit
print(submission_csv)

# submission_csv.to_csv(path + "sampleSubmission_0108_16.csv", index = False)

print("mse : ", loss)
y_predict=model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print("R2 스코어 : ", r2)
print("음수갯수 : ", submission_csv[submission_csv['count']<0].count())     # 데이터프레임 조건, 판다스 문법

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))     
rmse = RMSE(y_test, y_predict)             
print("mse : ", loss)                   
print("rmse", rmse)             

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

plt.title('바이크 로스')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()


'''
def RMSLE(y_test, y_predict):
    return np.sqrt(mean_squared_log_error(y_test, y_predict))     
rmsle = RMSLE(y_test, y_predict)             
print("rmsle", rmsle)             
'''

# validation
# 103/103 [==============================] - 0s 394us/step
# R2 스코어 :  0.30791699237131775
# 음수갯수 :  datetime    0
# count       0
# dtype: int64
# mse :  22483.3515625
# rmse 149.94447852866728