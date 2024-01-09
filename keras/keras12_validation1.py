# 06_1 카피
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5])
y_train = np.array([1,2,3,4,6])

x_val = np.array([6,7])     
y_val = np.array([5,7])


x_test = np.array([8,9,10])
y_test = np.array([8,9,10])

#.2 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 2, validation_data = (x_val, y_val))    # validation_data 검증

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11000, 7])    # x에 넣어주는 다음 예측값을 통해 다음 y값을 예측

print("로스 : ", loss)
print("[10] 의 예측값 : ", result)


# Epoch 1000/1000
# 4/4 [==============================] - 0s 0s/step - loss: 0.2826
# 1/1 [==============================] - 0s 67ms/step - loss: 0.0133         <-- 여기 : 훈련으로 인한 로스
# 1/1 [==============================] - 0s 74ms/step
# 로스 :  0.013271371833980083                                               <-- 여기 : 평가로 인한 로스                        
# [10] 의 예측값 :  [[1.0680186e+04]                                        ★ 훈련 로스값과 평가 로스값의 차이가 적은것이 과적합이 적다는 뜻
#  [6.9454489e+00]]