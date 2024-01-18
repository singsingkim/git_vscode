# https://dacon.io/competitions/open/236068/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time            

#1. 데이터

path = "c:/_data/dacon/cancer//"

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "diabetes_sub_0117_1.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
print(submission_csv)

print(train_csv.shape)          # (652, 9)
print(test_csv.shape)           # (116, 8) 아래 서브미션과의 열의 합이 12 인것은 id 열 이 중복되어서이다
print(submission_csv.shape)     # (116, 2)

print(train_csv.columns)        
# Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
#        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
#       dtype='object')

print(train_csv.info())
print(test_csv.info())
print(train_csv.describe())    # 평균,최소,최대 등등 표현 # DESCR 보다 많이 활용되는 함수. 함수는 () 붙여주어야 한다 이게 디폴트값

######### 결측치 처리 1. 제거 #########
train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
######### 결측치 처리 2. 0으로 #########
# train_csv = train_csv.fillna(0)   # 결측치 행에 0을 집어 넣는다
print(train_csv.isna().sum())       # 위 와 같다. isnull() = isna()
print(train_csv.info())
print(train_csv.shape)              # (652, 9)

test_csv = test_csv.fillna(test_csv.mean())     # 널값에 평균을 넣은거
print(test_csv.info())

######### x 와 y 를 분리 #########
x = train_csv.drop(['Outcome'], axis = 1)     # Outcome 를 삭제하는데 열이면 액시스 1, 행이면 0
y = train_csv['Outcome']

print(np.unique(y, return_counts=True))
# (array([0, 1], dtype=int64), array([424, 228], dtype=int64))

# 넘파이 갯수 함수
print(np.count_nonzero(y==0))
print(np.count_nonzero(y==1))
print(np.count_nonzero(y==2))
# 판다스 갯수 함수
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split( x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 2543)


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


print(x_train.shape, x_test.shape)  # (456, 8) (196, 8)
print(y_train.shape, y_test.shape)  # (456,) (196,)




#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 8))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dropout(0.3))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dropout(0.4))
model.add(Dense(1, activation = 'sigmoid'))     
# 시그모이드를 사용하면 0 ~ 1 사이의 값이 나온다.안그러면 0 ~ 1 바깥으로 값이 튄다
# 다중분류 일때는 카테고리 크로스엔트로피를 사용할때는 소프트맥스를 사용한다
# 이진분류 일때는 바이너리 크로스엔트로피를 사용할때는 시그모이드를 사용한다.


#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping,ModelCheckpoint       # 클래스는 정의가 필요
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date)         # 2024-01-17 10:54:10.769322
print(type(date))   # <class 'datetime.datetime')
date = date.strftime("%m%d_%H%M")
print(date)         # 0117_1058
print(type(date))   # <class 'str'>

path2='..\_data\_save\MCP\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
filepath = "".join([path2,'k28_07_dacon_diabetes_', date,'_', filename])
# '..\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
                     mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
                     patience=200,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
                     verbose=1,
                     restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
                     )

mcp = ModelCheckpoint(
    monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
    filepath=filepath
    )

model.compile(loss = 'binary_crossentropy', 
              optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
              metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.

hist = model.fit(x_train, y_train, epochs = 4000,
                 batch_size = 100, validation_split=0.2,
                 verbose=1, callbacks=[es,mcp])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = np.around(model.predict(x_test))
y_submit = np.around(model.predict(test_csv))
# y_submit = np.around(model.predict(x_test))
# r2 = r2_score(y_test, y_predict)

def ACC(aaa, bbb):
    return accuracy_score(aaa, bbb)
acc = ACC(y_test, y_predict)

print(y_submit.shape)  
print("========================================")
######## submission.csv 만들기(count 컬럼에 값만 넣어주면 됌) ########
submission_csv['Outcome'] = y_submit

submission_csv.to_csv(path + "sample_submission_0117_1.csv", index = False)

print("로스 : ", loss)
print("ACC : ", acc)

# 로스 :  [0.4606044590473175, 0.7806122303009033]
# ACC :  0.7806122448979592