from sklearn.datasets import fetch_covtype
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression # 로지스틱리그리션 = 분류다
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


datasets = fetch_covtype()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (581012, 54) (581012,)
print(pd.value_counts(y))
# 2    283301
# 1    211840
# 3     35754
# 7     20510
# 6     17367
# 5      9493
# 4      2747

# ★ 코드 완성 했을때 오류가 생길텐데 케라스 판다스 사이킷런 전처리과정에서 생길예정
# y의 시작값이 1부터 7까지의 6 개인 이유로 오류가 발생할것
# 오류가 나는 코드는 수정해서 완료시킬것

print(y)
print(np.unique(y, return_counts=True))
# (array([1, 2, 3, 4, 5, 6, 7]), array([211840, 283301,  35754,   2747,   9493,  17367,  20510], dtype=int64))

# ============================================
# ========== 원 핫 인코딩 전처리 ==============
# 1) 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)   # [0. 0. 0. ... 1. 0. 0.]
# print(y_ohe)
# print(y_ohe.shape)  # (581012, 8)

# # 2) 판다스
# y_ohe2 = pd.get_dummies(y)  # False  False  False  False   True  False  False
# print(y_ohe2)
# print(y_ohe2.shape) # (581012, 7)

# # 3) 사이킷런
# from sklearn.preprocessing import OneHotEncoder
# ohe = OneHotEncoder()   # (sparse=False)
# y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
# y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
# print(y_ohe3)
# print(y_ohe3.shape) # (581012, 7)


x_train, x_test, y_train, y_test = train_test_split(x, y, 
            shuffle=True, train_size= 0.7,
            random_state= 687415,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것


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

print(np.unique(y_test, return_counts=True))
# (array([0., 1.], dtype=float32), array([1220128,  174304], dtype=int64))


# 2. 모델구성
# model = LinearSVC(C = 100)
# model = Perceptron()
# model = LogisticRegression()
model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()



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

# path='c:\_data\_save\MCP\\'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
# filepath = "".join([path,'k30_09_fetch_covtype_', date,'_', filename])
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
#                  batch_size = 1000, validation_split=0.2,
#                  verbose=1, callbacks=[es,mcp])
# end_time=time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('results : ', results)
y_predict = model.predict(x_test)
acc = accuracy_score(y_predict, y_test)
print('acc : ', acc)
# results = model.evaluate(x_test, y_test)
# y_predict = model.predict(x_test)

# print(y_test)
# print(y_predict.shape, y_test.shape)    # (174304,) (174304,)

# y_test = np.argmax(y_test, axis=1)
# y_predict = np.argmax(y_predict, axis=1)
# print(y_test)
# print(y_predict)

# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_predict, y_test)
# print("로스 : ", results[0])
# print("ACC : ", results[1])
# print("accuracy_score : ", acc)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")

# # 로스 :  0.6851146817207336
# # ACC :  0.7185664176940918
# # accuracy_score :  0.7185664127042408