import numpy as np
import pandas as pd
from sklearn.datasets import load_iris  # 꽃
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time            

# 1. 데이터
datasets = load_iris()
# print(datasets) # 0 1 두가지면 이진-바이너리, 0 1 2 세개니까 다중-소프트맥스
# print(datasets.DESCR)   # 라벨 = Class 동일하다
print(datasets.feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

x = datasets.data
y = datasets.target
# print(x.shape, y.shape) # (150, 4) (150,)
print(y)
print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([50, 50, 50], dtype=int64))
print(pd.value_counts(y))   # y라벨의 갯수 확인하는 이유 : 하나가 너무 많으면 과적합이 뜬다
# 0    50                   # 어떤게 적다면 데이터 증폭으로 큰쪽에 맞춰주어야 한다 # 작은 쪽에 맞추려 삭제한다면 데이터 아깝. 또한 정확도 떨어짐
# 1    50
# 2    50

##### 맹그러봐 #####

# x_train, x_test, y_train, y_test = train_test_split(x, y, 
#             shuffle=True, train_size= 0.7,
#             random_state= 2543)

# print(x_train.shape, x_test.shape)  # (105, 4) (45, 4)
# print(y_train.shape, y_test.shape)  # (105,) (45,)
                                    # y를 (n, 3) 으로 one hot 형태로 바꺼주어야 한다
########### one hot 전처리 ############
# 1. keras    2. sklearn   3. pandas
# 1. 케라스
from keras.utils import to_categorical
y_ohe = to_categorical(y)
print(y_ohe)
print(y_ohe.shape)  # (150, 3)

# 2. 판다스
y_ohe2 = pd.get_dummies(y)
print(y_ohe2)
print(y_ohe2.shape) # (150, 3)


# 3. 사이킷런
print("============= 원핫 3 ===================")
# from sklearn.preprocessing import OneHotEncoder
# y = y.reshape(-3, 1)   # (150, 1) // 반장님 뉴스킬
# y = y.reshape(-1, 1)   # (150, 1) // 벡터를 행렬로 바꾼거.
# y = y.reshape(150, 1)  # (150, 1)
# 리쉐이프 중요한것 ★ 데이터가 바뀌엇는지, 데이터의 순서가 바뀌엇는지
# [[1,2,3,],[4,5,6,]] # (2, 3)
# [[1,2,3,4,5,6,]]    # (6, 0)
# [[1,2],[3,4],[5,6]] # (3, 2)
# ============== ★★★★★★★ ===================
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()#(sparse=False)           # 클래스 상태기 때문에 정의해준것
# ohe.fit(y)                    # 훈련을 시키지만 메모리 상태에서만 인식해준것
# y_ohe3 = ohe.transform(y)     # 핏 시켜 훈련시킨것을 트랜스폼을 통해 저장해준다
y = y.reshape(-1, 1)   # (150, 1) // 벡터를 행렬로 바꾼거.
# y_ohe3 = ohe.fit_transform(y)   # 위 두 줄과 결과가 같다 // fit + transform 과 같다
y_ohe3 = ohe.fit_transform(y).toarray()   # 위 두 줄과 결과가 같다 // fit + transform 과 같다

print(y_ohe3)
print(y_ohe3.shape)
# ============== ★★★★★★★ ===================


# print(y_ohe3.shape)
# enc = OneHotEncoder(sparse=False).fit(y_ohe3)
# y_ohe3 = enc.transform(y_ohe3)
# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것

print(np.unique(y_test, return_counts=True))

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 4))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3, activation = 'softmax'))     
# 시그모이드를 사용하면 0 ~ 1 사이의 값이 나온다.안그러면 0 ~ 1 바깥으로 값이 튄다
# 다중분류 일때는 카테고리 크로스엔트로피를 사용할때는 소프트맥스를 사용한다
# 이진분류 일때는 바이너리 크로스엔트로피를 사용할때는 시그모이드를 사용한다.

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
              metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.
# start_time = time.time()

from keras.callbacks import EarlyStopping       # 클래스는 정의가 필요
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
                     mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
                     patience=10,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
                     verbose=1,
                     restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
                     )

hist = model.fit(x_train, y_train, epochs = 10,
                 batch_size = 10, validation_split=0.2,
                 verbose=1, callbacks=[es])
# end_time = time.time()



#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

y_predict = model.predict(x_test)
print(y_predict)

print(y_test)
print(y_predict.shape, y_test.shape)

# y_test = y_test.reshape(90, )
# y_predict = y_predict.reshape(-1, )   # 하면 안되는 짓
# print(y_predict.shape, y_test.shape)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print(y_test)
print(y_predict)


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)

# 와이프레딕트에서 나온값을 np.아그맥스 를 통하면 숫자 비교 하고 가장 큰놈의 위치를 1로 잡겠다.
# 분류에서 애큐러스 스코어를 사용할떄 

# ★★★★ 시각화 ★★★★  
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

plt.title('iris')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

# ================케라스==================
# 로스 :  [0.15452216565608978, 0.9777777791023254] 두번째 값은 메트릭스에 넣은 acc 값
# ACC :  0.9777777777777777

# ================판다스==================
# 로스 :  [0.10661318898200989, 0.9555555582046509]
# ACC :  0.9555555555555556

# ================사이킷런==================
# 로스 :  [0.09952615946531296, 0.9555555582046509]
# ACC :  0.9555555555555556