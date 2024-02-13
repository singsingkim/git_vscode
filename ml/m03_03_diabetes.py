# 14_3 카피
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score    # r2 결정계수
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron, LogisticRegression # 리그리션 = 분류다
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
            train_size = 0.7,
            test_size = 0.3,
            shuffle = True,
            random_state = 4567)


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


print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

x = x.reshape(-1, 10, 1)


# 2 모델
# model = LinearSVC(C = 100)
# model = Perceptron()
# model = LogisticRegression()
# model = KNeighborsClassifier()
# model = DecisionTreeClassifier()
model = RandomForestClassifier()



#3. 컴파일, 훈련
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
# filepath = "".join([path,'k54_03_diabetes_', date,'_', filename])
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
#     )

# model.compile(loss = 'mse', optimizer = 'adam')
# start_time = time.time()
# hist = model.fit(x_train, y_train, epochs = 10, 
#             batch_size = 10, validation_split=0.3,
#             verbose=1, callbacks=[es,mcp]
#             )
# end_time = time.time()

#4. 평가, 예측
# loss = model.evaluate(x_test, y_test)
results = model.score(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)    # 결정계수

# def RMSE(aaa, bbb):
#     return np.sqrt(mean_squared_error(aaa, bbb))
# rmse = RMSE(y_test, y_predict)


# print(hist.history['val_loss'])

print("acc : ", results)
print("R2 스코어 : ", r2)
# print("RMSE : ", rmse)
# print("걸린 시간 : ", round(end_time - start_time,2),"초")


