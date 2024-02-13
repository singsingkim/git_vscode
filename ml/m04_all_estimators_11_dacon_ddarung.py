# https://dacon.io/competitions/open/235576/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    # rmse 사용자정의 하기 위해 불러오는것
import time                
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')



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
print(test_csv.shape)           # (715, 9) 아래 서브미션과의 열의 합이 11 인것은 id 열 이 중복되어서이다
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
print(train_csv)

test_csv = test_csv.fillna(test_csv.mean())     # 널값에 평균을 넣은거
print(test_csv.info())


######### x 와 y 를 분리 #########
x = train_csv.drop(['count'], axis = 1)     # count를 삭제하는데 count가 열이면 액시스 1, 행이면 0
print(x)
y = train_csv['count']
print(y)

x_train, x_test, y_train, y_test = train_test_split(
            x, y, shuffle=True, 
            train_size= 0.7, random_state= 45687
            )


from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.preprocessing import StandardScaler, RobustScaler


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(np.min(x_train))  # 
print(np.min(x_test))   # 
print(np.max(x_train))  # 
print(np.max(x_test))   # 

print(x_train.shape, x_test.shape)  # (929, 9) (399, 9)
print(y_train.shape, y_test.shape)  # (929,) (399,)

## 2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier')    # 분류
allAlgorithms = all_estimators(type_filter='regressor')   # 회귀

print('allAlgorithms', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms)) # 55 개 # 소괄호로 묶여 있으니 튜플
# 포문을 쓸수있는건 이터너리 데이터 (리스트, 튜플, 딕셔너리)
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

for name, algorithm in allAlgorithms:
    try:        
        # 2 모델
        model = algorithm()
        # 3 훈련
        model.fit(x_train, y_train)
        
        acc = model.score(x_test, y_test)
        print(name, '의 정답률 :', acc)
    except:
        print(name, ': 에러 입니다.')
        # continue # 문제가 생겼을때 다음 단계로


#3. 컴파일, 훈련
model.fit(x_train, y_train)
# from keras.callbacks import EarlyStopping, ModelCheckpoint       # 클래스는 정의가 필요
# import datetime
# date = datetime.datetime.now()
# print(date)         # 2024-01-17 10:54:10.769322
# print(type(date))   # <class 'datetime.datetime')
# date = date.strftime("%m%d_%H%M")
# print(date)         # 0117_1058
# print(type(date))   # <class 'str'>

# path2='c:\_data\_save\MCP\\'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
# filepath = "".join([path2,'k30_04_dacon_ddarung_', date,'_', filename])
# # 'c:\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#             mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#             patience=100,      # 최소값 찾은 후 설정값 만큼 훈련 진행
#             verbose=1,
#             restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
#             )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath=filepath
#     )

# model.compile(loss = 'mse', optimizer = 'adam')

# start_time = time.time()

# hist = model.fit(x_train, y_train, epochs = 10000,
#             batch_size = 100, validation_split=0.2,
#             verbose=1, callbacks=[es,mcp]
#             )
# end_time = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
print('acc :', results)

# acc : 0.7702005795147138


# print(y_submit)

# print(y_submit.shape)   # (715, 1)
# print("========================================")
# ######## submission.csv 만들기(count 컬럼에 값만 넣어주면 됌) ########
# submission_csv['count'] = y_submit
# print(submission_csv)

# submission_csv.to_csv(path + "ddarung_sub_0118_2.csv", index = False)

# y_predict=model.predict(x_test)
# r2 = r2_score(y_test, y_predict)

# print("로스 : ", loss)
# print("R2 스코어 : ", r2)

# def RMSE(aaa, bbb):
#     return np.sqrt(mean_squared_error(aaa, bbb))
# rmse = RMSE(y_test, y_predict)

# print(hist.history['val_loss'])

# print("로스 : ", loss)
# print("R2 스코어 : ", r2)
# print("RMSE : ", rmse)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")

# 로스 :  2889.401123046875
# R2 스코어 :  0.5646741422286308
# RMSE :  53.75314854187823

# 로스 :  2720.884033203125
# R2 스코어 :  0.6307659489319123
# RMSE :  52.16209301840004