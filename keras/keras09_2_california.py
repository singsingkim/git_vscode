from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
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

model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 2000, batch_size = 50)
end_time = time.time()

loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# epochs = 2000 , batch_size = 50 , random_state = 1

# Epoch 2000/2000
# 289/289 [==============================] - 0s 521us/step - loss: 0.5747
# 194/194 [==============================] - 0s 438us/step - loss: 0.5791
# 194/194 [==============================] - 0s 345us/step
# 645/645 [==============================] - 0s 389us/step
# 로스 :  0.5790699124336243
# R2 스코어 :  0.5595365853230052
# 걸린시간 :  306.56 초

