import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])   #range 는 기본적으로 python 에서 제공하는가서
x = x.T
print(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]) # 괄호안의 데이터 --> ★두개이상은 리스트
print("y 형태 : ", y.shape)
y = y.T

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 1000, batch_size = 2)

loss = model.evaluate(x, y)
result = model.predict([10])    # x에 넣어주는 다음 예측값을 통해 다음 y값을 예측

print("[10] 의 예측값 : ", result)

# 실습 : 만들기
# 예측 : [11, 2, -1]

# Epoch 1000/1000
# 5/5 [==============================] - 0s 0s/step - loss: 1.9211e-13
# 1/1 [==============================] - 0s 82ms/step - loss: 3.4441e-13
# 1/1 [==============================] - 0s 70ms/step
# [10] 의 예측값 :  [[11.000001    1.9999987  -0.99999905]]

