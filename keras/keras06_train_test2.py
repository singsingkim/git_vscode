import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

# [실습] 넘파이 리스트의 슬라이싱 --> 7:3 으로 분할
# 훈련 데이터와 평가 데이터를 분할하는 이유는 과적합 방지하기 위해
x_train = x[:7:1] # == [:-3]      # [1 2 3 4 5 6 7]  # 슬라이싱 구간 [시작값 : 도착값 : 간격값]
y_train = y[:7]   #       # [1 2 3 4 5 6 7]  # 간격값 생략시 기본 간격으로 1


'''
a = b   # a 라는 변수에 b 값을 넣어라
a == b  # a 와 b 가 같다

'''


x_test = x[7:10:1]      # [7:10]==  [-3:] == [-3 : 10] --> [ 8  9 10]
y_test = y[7:]          # [ 8  9 10]

print(x_train)
print(y_train)
print(x_test)
print(y_test)


#.2 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 2)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11000, 7])    # x에 넣어주는 다음 예측값을 통해 다음 y값을 예측

print("로스 : ", loss)
print("[10] 의 예측값 : ", result)


# Epoch 1000/1000
# 4/4 [==============================] - 0s 0s/step - loss: 1.1369e-13
# 1/1 [==============================] - 0s 72ms/step - loss: 3.0316e-12
# 1/1 [==============================] - 0s 69ms/step
# 로스 :  3.031648933629616e-12
# [10] 의 예측값 :  [[1.1000002e+04]
#  [6.9999995e+00]]

# ★ 훈련 로스값과 평가 로스값의 차이가 적은것이 과적합이 적다는 뜻