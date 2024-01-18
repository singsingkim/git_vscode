import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,)
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
print(y)

print(np.unique(y, return_counts=True))
# (array([0, 1, 2]), array([59, 71, 48], dtype=int64))
# 0 은 59개
# 1 은 71개
# 2 는 48개
print(pd.value_counts(y))
# 1    71
# 0    59
# 2    48
# ============================================
# ========== 원 핫 인코딩 전처리 ==============
# 1) 케라스
from keras.utils import to_categorical
y_ohe = to_categorical(y)   # [1. 0. 0. ] 으로 표현
print(y_ohe)
print(y_ohe.shape)  # (178, 3)

# 2) 판다스
y_ohe2 = pd.get_dummies(y)  # [True  False  False] 으로 표현
print(y_ohe2)
print(y_ohe2.shape) # (178, 3)

# 3) 사이킷런
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()   # (sparse=False)
y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
print(y_ohe3)
print(y_ohe3.shape) # (178, 3)

x_train, x_test, y_train, y_test = train_test_split(x, y_ohe, 
            shuffle=True, train_size= 0.7,
            random_state= 7777,
            stratify=y,)    # 스트레티파이 와이(예스)는 분류에서만 쓴다, 트레인 사이즈에 따라 줄여주는것


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





print(np.unique(y_test, return_counts=True))
# (array([0., 1.], dtype=float32), array([108,  54], dtype=int64))

print(x)
print(y)

#2. 모델구성
model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(3, activation = 'softmax'))     

#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
              metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.

from keras.callbacks import EarlyStopping, ModelCheckpoint       # 클래스는 정의가 필요
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
print(date)         # 2024-01-17 10:54:10.769322
print(type(date))   # <class 'datetime.datetime')
date = date.strftime("%m%d_%H%M")
print(date)         # 0117_1058
print(type(date))   # <class 'str'>

path='..\_data\_save\MCP\\'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
filepath = "".join([path,'k26_08_wine_', date,'_', filename])
# '..\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
                     mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
                     patience=100,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
                     verbose=1,
                     restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
                     )
mcp = ModelCheckpoint(
    monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
    filepath=filepath
    )

hist = model.fit(x_train, y_train, epochs = 1000,
                 batch_size = 10, validation_split=0.2,
                 verbose=1, callbacks=[es,mcp])

model.save_weights("..\_data\_save\keras26_8_save_weights.h5")


#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("로스 : ", results[0])
print("ACC : ", results[1])

print(y_test)
print(y_predict.shape, y_test.shape)    # (54, 3) (54, 3)

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)
print(y_test)
print(y_predict)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_predict, y_test)
print("accuracy_score : ", acc)
print(y_predict)

# 와이프레딕트에서 나온값을 np.아그맥스 를 통하면 숫자 비교 하고 가장 큰놈의 위치를 1로 잡겠다.
# 분류에서 애큐러스 스코어를 사용할떄 

# accuracy_score :  0.8888888888888888


print(x.shape, y.shape)
print(x_train.shape, y_train.shape)

print("로스 : ", results[0])
print("ACC : ", results[1])



# 민맥스스케일
# 로스 :  0.04872474446892738
# ACC :  0.9814814925193787

# 맥스앱스스케일
# 로스 :  0.06441733241081238
# ACC :  0.9629629850387573

# 스탠다드스케일
# 로스 :  0.11139846593141556
# ACC :  0.9814814925193787

# 로부투스스케일
# 로스 :  0.13515536487102509
# ACC :  0.9814814925193787
