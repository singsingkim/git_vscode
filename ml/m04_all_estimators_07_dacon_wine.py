# https://dacon.io/competitions/open/235610/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error    # rmse 사용자정의 하기 위해 불러오는것
import time                
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


#1. 데이터

path = "c:/_data/dacon/wine//"

# print(path + "aaa.csv") # c:/_data/dacon/ddarung/aaa.csv

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
# \ \\ / // 다 가능
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

print(train_csv.shape)          # (5497, 13)
print(test_csv.shape)           # (1000, 12) // 아래 서브미션과의 열의 합이 11 인것은 id 열 이 중복되어서이다
print(submission_csv.shape)     # (1000, 2)

print(train_csv.columns)        # 


print(train_csv.info())
print("==========================")
print(test_csv.info())
print(train_csv.describe())

######### 결측치 처리 1. 제거 #########
train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
######### 결측치 처리 2. 0으로 #########
# train_csv = train_csv.fillna(0)   # 결측치 행에 0을 집어 넣는다

# print(train_csv.isnull().sum())
print(train_csv.isna().sum())       # 위 와 같다. isnull() = isna()
print(train_csv.info())
print(train_csv.shape)
print(train_csv)

# test_csv = test_csv.fillna(test_csv.mean())     # 널값에 평균을 넣은거
print(test_csv.info())

# type에는 white와 red 두 종류가 있습니다.
# 각각 0,1로 변환합니다.
train_csv['type'] = train_csv['type'].map({'white':0, 'red':1}).astype(int)
test_csv['type'] = test_csv['type'].map({'white':0, 'red':1}).astype(int)


######### x 와 y 를 분리 #########
x = train_csv.drop(['quality'], axis = 1)     # species 를 삭제하는데 count가 열이면 액시스 1, 행이면 0
y = train_csv['quality']

print(np.unique(y, return_counts=True))
# (array([3, 4, 5, 6, 7, 8, 9], dtype=int64), array([  26,  186, 1788, 2416,  924,  152,    5], dtype=int64))




print(pd.value_counts(y))
# 6    2416
# 5    1788
# 7     924
# 4     186
# 8     152
# 3      26
# 9       5


# ========== 원 핫 인코딩 전처리 ==============
# 1) 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)   # [1. 0. 0. ] 으로 표현
# print(y_ohe)
# print(y_ohe.shape)  # (5497, 10)

# # 슬라이싱해서 0번째를 자름 / 7 을 0으로 바꿀수 있다 - 라벨값이 평등 / 0부터 라벨링이 줄어듬
# # 최대값에서 플러스 일 만큼 만들어줌

# # 2) 판다스
# y_ohe2 = pd.get_dummies(y)  # [True  False  False] 으로 표현
# print(y_ohe2)
# print(y_ohe2.shape) # (5497, 7)


# '''
# # 3) 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()   # (sparse=False)
# y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
# y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
# print(y_ohe3)
# print(y_ohe3.shape) # (178, 3)
# '''


x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 78567,
            stratify=y)    # 스트레티파이 y_ohe2


# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
# from sklearn.preprocessing import StandardScaler, RobustScaler


# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 



# print(np.unique(y_test, return_counts=True))
# # (array([False,  True]), array([9900, 1650], dtype=int64))
# print(x, y)
# print(x_train.shape, x_test.shape)  # ((3847, 12) (1650, 12)
# print(y_train.shape, y_test.shape)  # ((3847, 7) (1650, 7)


## 2. 모델구성
allAlgorithms = all_estimators(type_filter='classifier')    # 분류
# allAlgorithms = all_estimators(type_filter='regressor')   # 회귀

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
# model.compile(loss = 'categorical_crossentropy', 
#               optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
#               metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.

# from keras.callbacks import EarlyStopping,ModelCheckpoint       # 클래스는 정의가 필요
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# import datetime
# date = datetime.datetime.now()
# print(date)         # 2024-01-17 10:54:10.769322
# print(type(date))   # <class 'datetime.datetime')
# date = date.strftime("%m%d_%H%M")
# print(date)         # 0117_1058
# print(type(date))   # <class 'str'>

# path2='c:\_data\_save\MCP\\'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
# filepath = "".join([path2,'k30_10_dacon_wine_', date,'_', filename])
# # 'c:\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#                      mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#                      patience=100,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
#                      verbose=1,
#                      restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
#                      )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath=filepath
#     )
# start_time=time.time()
# hist = model.fit(x_train, y_train, epochs = 10000,
#                  batch_size = 20, validation_split=0.2,
#                  verbose=1, callbacks=[es,mcp], )
# end_time=time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('results : ', results)

# results = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)
# print("로스 : ", results[0])
# print("ACC : ", results[1])
# print(y_predict)


# print(y_test)
# print(y_predict.shape, y_test.shape)    # 

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)


# # ============= 모델을 최종적으로 완성 후 테스트값 받은 파일을 모델 돌려서 예측값을 서브밋에 뽑아낸 것===
# y_submit = model.predict(test_csv)  # 테스트 파일을 모델에 넣어서 예측값을 뽑아내서. 와이서브밋에 저장한거를 아래 서브미션 씨에스브이에서 저장
# # ================================================

# # y_submit = np.arg max(y_submit, axis=1)


# print(y_test)
# print(y_predict)
# y_submit=np.argmax(y_submit, axis=1)+3
# ######## submission.csv 만들기(species 컬럼에 값만 넣어주면 됌) ########
# submission_csv['quality'] = y_submit
# # submission_csv['quality'] = np.argmax(y_submit)+3

# #submission_csv['quality']= y_submit

# print(submission_csv)



# # ============= 모델을 위에서 뽑아낸 것을 csv 파일로 생성===============
# submission_csv.to_csv(path + "wine_sub_0118_1.csv", index = False)


# from sklearn.metrics import accuracy_score
# # acc = accuracy_score(y_predict, y_test)
# # print("accuracy_score : ", acc)


# def ACC(a,b):
#     return accuracy_score(a,b)
# acc = ACC(y_test, y_predict)

# print("로스 : ", results[0])
# print("ACC : ", results[1])
# print("acc : ", acc)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")


# # 로스 :  1.0892001390457153
# # ACC :  0.5345454812049866
# # acc :  0.5345454545454545


# results :  0.6678787878787878