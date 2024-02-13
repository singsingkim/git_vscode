# https://dacon.io/competitions/open/235576/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time                
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


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
                 x, y, shuffle=True, train_size= 0.7, 
                 random_state= 77777
)


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


print(x_train.shape, x_test.shape)  # (7620, 8) (3266, 8)
print(y_train.shape, y_test.shape)  # (7620,) (3266,)


## 2. 모델구성
# allAlgorithms = all_estimators(type_filter='classifier')    # 분류
allAlgorithms = all_estimators(type_filter='regressor')   # 회귀

print('allAlgorithms', allAlgorithms)
print('모델의 갯수 :', len(allAlgorithms)) # 41 개 # 소괄호로 묶여 있으니 튜플
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
# from keras.callbacks import EarlyStopping,ModelCheckpoint       # 클래스는 정의가 필요
# import datetime
# date = datetime.datetime.now()
# print(date)         # 2024-01-17 10:54:10.769322
# print(type(date))   # <class 'datetime.datetime')
# date = date.strftime("%m%d_%H%M")
# print(date)         # 0117_1058
# print(type(date))   # <class 'str'>

# path2='c:\_data\_save\MCP\\'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
# filepath = "".join([path2,'k30_05_kaggle_bike_', date,'_', filename])
# # 'c:\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#             mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#             patience=200,      # 최소값 찾은 후 열 번 훈련 진행
#             verbose=1,
#             restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
#             )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath=filepath
#     )

# model.compile(loss = 'mse', optimizer = 'adam')
# start_time=time.time()
# hist = model.fit(x_train, y_train, epochs = 10000,
#             batch_size = 40,validation_split= 0.3, 
#             verbose=1, callbacks=[es,mcp]
#             )
# end_time=time.time()

#4. 평가, 예측

# loss = model.evaluate(x_test, y_test)
results = model.score(x_test, y_test)
print('acc : ', results)

y_predict = model.predict(x_test)
print(y_predict)
# acc2 = accuracy_score(y_predict, y_test)
# print('acc2 : ', acc2)




# y_submit[y_submit < 0] = 0      # 결과가 나온값에 후처리를 한 것[프레딧 한 값이 그대로 나와야 하는데 이렇게하면 좋지 않다]

# print(y_submit)
# print(y_submit.shape)   # (6493, 1)
# print("========================================")
# ######## submission.csv 만들기(count 컬럼에 값만 넣어주면 됌) ########
# submission_csv['count'] = y_submit
# print(submission_csv)

# # 해당 경로에 submission_csv 파일 생성
# submission_csv.to_csv(path + "bike_sub_0118_1.csv", index = False)

# print("mse : ", loss)
# y_predict=model.predict(x_test)
# r2 = r2_score(y_test, y_predict)

# print("음수갯수 : ", submission_csv[submission_csv['count']<0].count())     # 데이터프레임 조건, 판다스 문법

# def RMSE(y_test, y_predict):
#     return np.sqrt(mean_squared_error(y_test, y_predict))     
# rmse = RMSE(y_test, y_predict)             

# print(hist.history['val_loss'])

# print("mse : ", loss)    
# print("R2 스코어 : ", r2)               
# print("rmse", rmse)  
# print("걸린 시간 : ", round(end_time - start_time,2),"초")

# mse :  26685.71484375
# R2 스코어 :  0.19420721353364867
# rmse 163.35764540289156