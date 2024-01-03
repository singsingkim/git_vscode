import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([range(10)])   #range 는 기본적으로 python 에서 제공하는가서
                            #range 는 10가지의 숫자. 0~9
print(x)
print(x.shape)    # (1, 10)

x = np.array([range(1, 10)])
print(x)        # [[1 2 3 4 5 6 7 8 9]]
print(x.shape)  # (1, 9)

x = np.array([range(10), range(21, 31), range(201, 211)])
print(x)
print(x.shape)
x = x.T
print(x)
print(x.shape)

y = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9],
             [9,8,7,6,5,4,3,2,1,0]]) # 괄호안의 데이터 --> ★두개이상은 리스트
print("y 형태 : ", y.shape)
y = y.T

model = Sequential()
model.add(Dense(1, input_dim = 3))
model.add(Dense(1))
model.add(Dense(4))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(3))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 3000, batch_size = 2)

loss = model.evaluate(x, y)
result = model.predict([[10, 31, 211]])

print("[10, 31, 211] 의 예측값 : ", result)

# 실습 : 만들기
# 예측 : [10, 31, 211]

# Epoch 3000/3000
# 5/5 [==============================] - 0s 0s/step - loss: 1.0123e-12
# 1/1 [==============================] - 0s 76ms/step - loss: 1.6276e-12
# 1/1 [==============================] - 0s 56ms/step
# [10, 31, 211] 의 예측값 :  [[11.000001   2.0000005 -1.0000032]]