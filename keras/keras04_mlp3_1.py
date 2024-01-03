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
             [1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9]]) # 괄호안의 데이터 --> ★두개이상은 리스트
print("y 형태 : ", y.shape)
y = y.T

model = Sequential()
model.add(Dense(1, input_dim = 3))  
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(2))

model.compile(loss ='mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size = 2)

loss = model.evaluate(x, y)
result = model.predict([[10, 31, 211]])     # y의 다음값 11, 2 를 예측      

print("[10, 31, 211]의 예측값 : ", result)
# 실습 : 만들기
# 예측 : [10, 31, 211]

# [[  0  21 201]
#  [  1  22 202]
#  [  2  23 203]
#  [  3  24 204]
#  [  4  25 205]
#  [  5  26 206]
#  [  6  27 207]
#  [  7  28 208]
#  [  8  29 209]
#  [  9  30 210]]  --> 프레딧에 다음 예측값인 10 , 31, 211
# [[ 1.   1. ]
#  [ 2.   1.1]
#  [ 3.   1.2]
#  [ 4.   1.3]
#  [ 5.   1.4]
#  [ 6.   1.5]
#  [ 7.   1.6]
#  [ 8.   1.7]
#  [ 9.   1.8]
#  [10.   1.9]]  --> 예측 원하는 값은 11 , 2


# Epoch 1000/1000
# 5/5 [==============================] - 0s 4ms/step - loss: 1.2379e-11
# 1/1 [==============================] - 0s 78ms/step - loss: 1.1998e-11
# 1/1 [==============================] - 0s 66ms/step
# [10, 31, 211]의 예측값 :  [[10.99999    2.0000026]]

