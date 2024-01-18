# multi layer perseptron
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Dropout

import tensorflow as ts
import keras

# 1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,  1.5, 1.4, 1.3]]
            )



y = np.array([1,2,3,4,5,6,7,8,9,10])


# x = x.T         # x.transpose() 동일         # 행과 열을 전치
# [[1,1], [2,1.1], [3,1.2], ... [10,1.3]]

# 2. 모델구성 순차적
# model = Sequential()
# model.add(Dense(10, input_shape = (2, )))  
# model.add(Dense(9, input_dim=10))
# model.add(Dropout(0.2))
# model.add(Dense(8))
# model.add(Dense(7))
# model.add(Dense(1))

# 2. 모델구성 함수형
input1 = Input(shape=(2,))
dense1 = Dense(10)(input1)
dense2 = Dense(9)(dense1)
drop1 = Dropout(0.2)(dense2)
dense3 = Dense(8, activation='relu')(drop1)
dense4 = Dense(7)(dense3)
output1 = Dense(1)(dense4)
model = Model(inputs=input1, outputs=output1)

model.summary()

# # 3. 컴파일, 훈련
# model.compile(loss ='mse', optimizer='adam')
# model.fit(x, y, epochs = 1000, batch_size = 2)

# # 4. 평가, 예측
# loss = model.evaluate(x, y)
# result = model.predict([[10, 1.3]])     # predict([10, 1.3]) --> 형태가 (2, ) 인 스칼라 0차원 형태이다.
#                                         # predict([[10, 1.3]]) --> 형태를 (1, 2) 1행 2열 벡터 형태인 1차원으로 만들어 주어야한다.
#                                         # (100, 3) 이나 (200, 3) 등등 (N, 3) 엔 콤마 3 열 형태로 인식해야 한다.

# print("[10, 1.3]의 예측값 : ", result)