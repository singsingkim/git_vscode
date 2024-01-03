import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,9,10])
y = np.array([1,2,3,4,6,5,7,9,10])

#.2 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 1000, batch_size = 2)

loss = model.evaluate(x, y)
result = model.predict([11000, 7])    # x에 넣어주는 다음 예측값을 통해 다음 y값을 예측

print("[10] 의 예측값 : ", result)


# Epoch 1000/1000
# 5/5 [==============================] - 0s 0s/step - loss: 0.2425
# 1/1 [==============================] - 0s 77ms/step - loss: 0.2318
# 1/1 [==============================] - 0s 82ms/step
# [10] 의 예측값 :  [[1.1035966e+04]
#  [7.0993819e+00]]