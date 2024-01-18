# 14_2 카피
from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (20640, 8) (20640,)
print(datasets.feature_names)
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)   # 20640 행 , 8 열

# [실습] 만들기
# R2 0.55 ~ 0.6 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 1)

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

# scaler = RobustScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# print(np.min(x_train))  # 
# print(np.min(x_test))   # 
# print(np.max(x_train))  # 
# print(np.max(x_test))   # 

# model = Sequential()
# model.add(Dense(64, input_dim = 8))
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(4))
# model.add(Dense(1))

# model.compile(loss = 'mse', optimizer = 'adam')
# start_time = time.time()


# 3
# from keras.callbacks import EarlyStopping, ModelCheckpoint       # 클래스는 정의가 필요
# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#             mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#             patience=30,      # 최소값 찾은 후 열 번 훈련 진행
#             verbose=1,
#             restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
#             )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath='..\_data\_save\MCP\keras26_2_MCP.hdf5'
# )

# hist = model.fit(x_train, y_train, epochs = 100, 
#             batch_size=50, validation_split=0.3, 
#             verbose=1, callbacks=[es]
#             )

model=load_model('..\_data\_save\MCP\keras26_2_MCP.hdf5')


# end_time = time.time()

# 4
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 2),"초")

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa,bbb))
rmse = RMSE(y_test, y_predict)

print("RMSE : ", rmse)
# print("걸린시간 : ", round(end_time - start_time, 2),"초")

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

print("로스 : ", loss)
print("R2 스코어 : ", r2)
# print("걸린시간 : ", round(end_time - start_time, 2),"초")

'''
# ★ 시각화 ★
import matplotlib.pyplot as plt
# from matplotlib import font_manager, rc
import matplotlib.font_manager as fm
font_path = "c:\Windows\Fonts\MALGUN.TTF"
font_name=fm.FontProperties(fname=font_path).get_name()
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.title('캘리포니아 로스')        # 한글깨짐 해결할것
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()
'''

# 민맥스스케일
# 로스 :  0.5334963798522949
# R2 스코어 :  0.5942014451217451
# 걸린시간 :  6.69 초

# 맥스앱스스케일
# 로스 :  0.5637744069099426
# R2 스코어 :  0.5711709946966239
# 걸린시간 :  16.37 초

# 스탠다드스케일
# 로스 :  0.5310338139533997
# R2 스코어 :  0.5960745867565976
# 걸린시간 :  10.12 초

# 로부투스스케일
# 로스 :  0.5331420302391052
# R2 스코어 :  0.5944709987126584
# 걸린시간 :  16.77 초