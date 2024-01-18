# https://dacon.io/competitions/open/236068/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time            

#1. 데이터

path = "c:/_data/dacon/cancer//"

train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
print(train_csv)
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
print(test_csv)
submission_csv = pd.read_csv(path + "sample_submission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.
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




# #2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim = 8))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1, activation = 'sigmoid'))     
# # 시그모이드를 사용하면 0 ~ 1 사이의 값이 나온다.안그러면 0 ~ 1 바깥으로 값이 튄다
# # 다중분류 일때는 카테고리 크로스엔트로피를 사용할때는 소프트맥스를 사용한다
# # 이진분류 일때는 바이너리 크로스엔트로피를 사용할때는 시그모이드를 사용한다.


# #3. 컴파일, 훈련
# model.compile(loss = 'binary_crossentropy', 
#               optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
#               metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.
# # start_time = time.time()

# from keras.callbacks import EarlyStopping,ModelCheckpoint       # 클래스는 정의가 필요
# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#                      mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#                      patience=200,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
#                      verbose=1,
#                      restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
#                      )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath='..\_data\_save\MCP\keras26_7_MCP.hdf5'
#     )



# hist = model.fit(x_train, y_train, epochs = 4000,
#                  batch_size = 100, validation_split=0.2,
#                  verbose=1, callbacks=[es,mcp])
# model.save_weights("..\_data\_save\keras26_7_save_weights.h5")


# # end_time = time.time()

model = load_model('..\_data\_save\MCP\keras26_7_MCP.hdf5')

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

submission_csv.to_csv(path + "sample_submission_0116_1.csv", index = False)

# print("============ hist =============")
# print(hist)
# print("===============================")
# # print(datasets) #데이터.데이터 -> 데이터.타겟
# print(hist.history)     # 오늘과제 : 리스트, 딕셔너리=키(loss) : 똔똔 밸류 한 쌍괄호{}, 튜플
#                                     # 두 개 이상은 리스트
#                                     # 딕셔너리
# print("============ loss =============")
# print(hist.history['loss'])
# print("============ val_loss =========")
# print(hist.history['val_loss'])
# print("===============================")

# print("걸린시간 : ", round(end_time - start_time, 2),"초")
print("로스 : ", loss)
# print("R2 스코어 : ", r2)
print("ACC : ", acc)

'''
# ★ 시각화 ★  
import matplotlib.pyplot as plt
# from matplotlib import font_manager
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['acc'], c='pink', label='acc', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
# plt.rcParams['font.family'] ='Malgun Gothic'
# plt.rcParams['axes.unicode_minus'] =False

plt.title('cancer')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

# random_state = 77777
# 로스 :  [0.4246818721294403, 0.8520408272743225]
# ACC :  0.8520408163265306


# random_state = 77777
# 로스 :  [0.43093279004096985, 0.8469387888908386]
# ACC :  0.8469387755102041
# 데이콘 점수 0.7931034483

# random_state = 254
# 로스 :  [0.5659946799278259, 0.75]
# ACC :  0.75
# 데이콘 점수 0.8


# random_state= 2543)
# 로스 :  [0.48043733835220337, 0.7857142686843872]
# ACC :  0.7857142857142857

'''

# 민맥스스케일
# 로스 :  [0.47522827982902527, 0.7857142686843872]
# ACC :  0.7857142857142857

# 맥스앱스스케일
# 로스 :  [0.4747494161128998, 0.7857142686843872]
# ACC :  0.7857142857142857

# 스탠다드스케일
# 로스 :  [0.4747494161128998, 0.7857142686843872]
# ACC :  0.7857142857142857

# 로부투스스케일
# 로스 :  [0.4651394784450531, 0.7704081535339355]
# ACC :  0.7704081632653061