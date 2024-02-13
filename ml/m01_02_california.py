# 14_2 카피
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv1D, Flatten, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time
from sklearn.svm import LinearSVC

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
            random_state = 4567)

print(x_train.shape, y_train.shape) # (14447, 8) (14447,)
# x_train = x_train.reshape(-1,8,1)
# x_test = x_test.reshape(-1,8,1)

# 2 모델
model = LinearSVC(C = 100)
# model = Sequential()
# model.add(Conv1D(256, 2, input_shape = (8, 1))) 
# model.add(Flatten()) #평탄화
# model.add(Dense(16))
# model.add(Dense(128))
# model.add(Dense(32))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(1))

# model.summary()



# # 3 컴파일, 훈련
model.fit(x_train, y_train)
# from keras.callbacks import EarlyStopping, ModelCheckpoint       # 클래스는 정의가 필요
# import datetime
# date = datetime.datetime.now()
# print(date)         # 2024-01-17 10:54:10.769322
# print(type(date))   # <class 'datetime.datetime')
# date = date.strftime("%m%d_%H%M")
# print(date)         # 0117_1058
# print(type(date))   # <class 'str'>

# path='c:\_data\_save\MCP\\'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'    # 0~9999 에포 , 0.9999 발로스
# filepath = "".join([path,'k54_02_california_', date,'_', filename])
# # 'c:\_data\_save\MCP\\k25_0117_1058_0101-0.3333.hdf5'

# es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
#             mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
#             patience=100,      # 최소값 찾은 후 열 번 훈련 진행
#             verbose=1,
#             restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
#             )

# mcp = ModelCheckpoint(
#     monitor='val_loss', mode = 'auto', verbose=1,save_best_only=True,
#     filepath=filepath
# )

# model.compile(loss = 'mse', optimizer = 'adam')
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs = 10, 
#             batch_size=50, validation_split=0.2, 
#             verbose=1, callbacks=[es, mcp]
#             )
# end_time = time.time()

# 4
# loss = model.evaluate(x_test,y_test)
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print("로스 : ", results)
print("R2 스코어 : ", r2)

# def RMSE(aaa, bbb):
#     return np.sqrt(mean_squared_error(aaa,bbb))
# rmse = RMSE(y_test, y_predict)

# print("RMSE : ", rmse)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")

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

# print("로스 : ", loss)
# print("R2 스코어 : ", r2)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")


# # # ★ 시각화 ★
# # import matplotlib.pyplot as plt
# # # from matplotlib import font_manager, rc
# # import matplotlib.font_manager as fm
# # font_path = "c:\Windows\Fonts\MALGUN.TTF"
# # font_name=fm.FontProperties(fname=font_path).get_name()
# # plt.figure(figsize=(9, 6))
# # plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
# # plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
# # plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# # # font_path = "C:/Windows/Fonts/NGULIM.TTF"
# # # font = font_manager.FontProperties(fname=font_path).get_name()
# # # rc('font', family=font)

# # plt.rcParams['font.family'] ='Malgun Gothic'
# # plt.rcParams['axes.unicode_minus'] =False

# # plt.title('캘리포니아 로스')        # 한글깨짐 해결할것
# # plt.xlabel('epoch')
# # plt.ylabel('loss')
# # plt.grid()
# # plt.show()


# # 로스 :  0.5326296091079712
# # R2 스코어 :  0.594860859824481


