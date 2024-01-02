import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as ts
import keras

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,  1.5, 1.4, 1.3]]
            )



y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)  # (2, 10)
print(y.shape)  # (10, ) --> 스칼라 10개짜리 한 묶음 --> 열의 개수는 항상 동일 

x = x.T         # x.transpose() 동일         # 행과 열을 전치
# [[1,1], [2,1.1], [3,1.2], ... [10,1.3]]
print(x.shape)  # (10, 2)

# 2. 모델구성
model = Sequential()
model.add(Dense(1, input_dim = 2))      # 디맨션이 2개란 의미. 아래 # 4 의 예측값도 열의 형태로 변해주어야함
# 열, 컬럼, 속성, 특성, 차원 = 2 // 같다
# (행무시, 열우선) <= 암기
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size = 2)

# 4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[10, 1.3]])     # predict([10, 1.3]) --> 형태가 (2, ) 인 스칼라 0차원 형태이다.
                                        # predict([[10, 1.3]]) --> 형태를 (1, 2) 1행 2열 벡터 형태인 1차원으로 만들어 주어야한다.
                                        # (100, 3) 이나 (200, 3) 등등 (N, 3) 엔 콤마 3 열 형태로 인식해야 한다.

print("[10, 1.3]의 예측값 : ", result)
# [실습] : 소수 2째 자리 까지 맞추기


# Epoch 1000/1000
# 5/5 [==============================] - 0s 0s/step - loss: 2.2455e-05
# 1/1 [==============================] - 0s 78ms/step - loss: 1.8709e-05
# 1/1 [==============================] - 0s 67ms/step
# [10, 1.3]의 예측값 :  [[10.004623]]