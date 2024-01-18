import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential,load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_squared_log_error, accuracy_score    # rmse 사용자정의 하기 위해 불러오는것
import time        
import numpy as np
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
# print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (569, 30) (569,)

print(np.unique(y, return_counts=True))
# (array([0, 1]), array([212, 357], dtype=int64))

# 넘파이 갯수 함수
print(np.count_nonzero(y==0))
print(np.count_nonzero(y==1))
print(np.count_nonzero(y==2))
# 판다스 갯수 함수
print(pd.DataFrame(y).value_counts())
print(pd.Series(y).value_counts())
print(pd.value_counts(y))

x_train, x_test, y_train, y_test = train_test_split(
                 x, y, shuffle=True, train_size= 0.7, 
                 random_state= 88888
                 )


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

# #2. 모델구성
# model = Sequential()
# model.add(Dense(64, input_dim = 30))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1, activation = 'sigmoid'))    # 시그모이드를 사용하면 0 ~ 1 사이의 값이 나온다

# # 분류에서는 mse 를 사용하지 않는다
# model.compile(loss='binary_crossentropy', optimizer='adam',
#               metrics=['acc','mse','mae']) # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.
# start_time = time.time()

# from keras.callbacks import EarlyStopping,ModelCheckpoint       # 클래스는 정의가 필요
# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#             mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#             patience=20,      # 최소값 찾은 후 열 번 훈련 진행
#             verbose=1,
#             restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
#             )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath='..\_data\_save\MCP\keras26_6_MCP.hdf5'
#     )

# hist = model.fit(x_train, y_train, epochs=300, 
#             batch_size=1, validation_split=0.2, 
#             verbose=1, callbacks=[es,mcp]     # 리스트 형태임 --> 친구가 있다
#             )
# model.save_weights("..\_data\_save\keras26_6_save_weights.h5")

# end_time = time.time()
model = load_model('..\_data\_save\MCP\keras26_6_MCP.hdf5')
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   # 엑스테스트를 프레딕트 시켜서 결과를 와이테스트와 비교함. 그 결과
y_predict = np.around(model.predict(x_test))
result = model.predict(x)

r2 = r2_score(y_test, y_predict)

### 해결 ### y_predict 에 소수점이 들어가있기 때문에 반올림 처리를 해줘야 y_test 0, 1 값과 일치를 시켜주어야 한다. 


def ACC(aaa, bbb):
    return accuracy_score(aaa, bbb)
acc = ACC(y_test, y_predict)


# print("걸린시간 : ", round(end_time - start_time, 2),"초")
print("로스 : ", loss)
print("R2 스코어 : ", r2)   # 분류에서는 신뢰할수 없는 지표. mse mae 또한 마찬가지(회귀 전용 지표)
print("ACC : ", acc)
# # print("============ hist =============")
# print(hist)
# # print("===============================")
# # print(datasets) #데이터.데이터 -> 데이터.타겟
# print(hist.history)     # 오늘과제 : 리스트, 딕셔너리=키(loss) : 똔똔 밸류 한 쌍괄호{}, 튜플
#                                     # 두 개 이상은 리스트
#                                     # 딕셔너리
# # print("============ loss =============")
# print(hist.history['loss'])
# # print("============ val_loss =========")
# print(hist.history['val_loss'])
# # print("===============================")

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

'''
# 걸린시간 :  49.02 초
# 로스 :  [0.07586187869310379, 0.9707602262496948, 0.022004911676049232, 0.05507449805736542]
# R2 스코어 :  0.8743386243386243
# ACC :  0.9707602339181286

# 민맥스스케일
# mse :  22164.369140625
# R2 스코어 :  0.3307323484759217
# rmse 148.87702937453156
# 걸린시간 :  48.75 초

# 맥스앱스스케일
# 걸린시간 :  11.09 초
# 로스 :  [0.13613742589950562, 0.9415204524993896, 0.040049958974123, 0.06330782175064087]
# R2 스코어 :  0.7486772486772486
# ACC :  0.9415204678362573

# 스탠다드스케일
# 걸린시간 :  11.09 초
# 로스 :  [0.13613742589950562, 0.9415204524993896, 0.040049958974123, 0.06330782175064087]
# R2 스코어 :  0.7486772486772486
# ACC :  0.9415204678362573

# 로부스터스케일
# 걸린시간 :  5.77 초
# 로스 :  [0.07427449524402618, 0.9707602262496948, 0.021223055198788643, 0.04237585887312889]
# R2 스코어 :  0.8743386243386243
# ACC :  0.9707602339181286
