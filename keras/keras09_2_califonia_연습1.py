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
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    test_size = 0.3,
                                                    shuffle = True,
                                                    random_state = 1)
print(x.shape, y.shape)         # (20640, 8) (20640,)
print(datasets.feature_names)   # ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
print(datasets.DESCR)           # 20640 행 , 8열

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 8))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 1000, batch_size = 100)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)
r2 = r2_score(y_test, y_predict)

print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린 시간 : ", round(end_time - start_time, 2), "초")




