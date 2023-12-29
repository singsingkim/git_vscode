import tensorflow as tf # tensorflow를 땡겨오고, tf 라고 줄여서 쓴다
print(tf.__version__)   # 2.15.0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense #
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])
import numpy as np

#2.모델구성 y = wx + b
model = Sequential() # 모델이라는 변수, 
model.add(Dense(1, input_dim=1))    # 인풋딤= x 한덩어리,1= y 한덩어리를 엮을거다

#3. 컴파일, 훈련량 # 최적의 weight # 최소의 loss
model.compile(loss='mse', optimizer='adam')     # mse로스값은 항상 양수(제곱해서 양수로 만들고 루트), Adam 은 일단 외우기(성능향상) 
model.fit(x, y, epochs=10000)       # fit 은 훈련하라. 훈련 한 번 하면 랜덤값. 두 번 하면 더 로스 적은 값. 
                                    #훈련을 너무 많이하게 되면 과적합 뜰 수 있음.--> 훈련량 조절 필요(epochs) --> 최적의 웨이트 생성

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([4])
print("4의 예측값은 : ", result)

# Epoch 10000/10000
# 1/1 [==============================] - 0s 2ms/step - loss: 3.4106e-13
# 1/1 [==============================] - 0s 59ms/step - loss: 3.4106e-13
# 로스 :  3.410605131648481e-13
# 1/1 [==============================] - 0s 39ms/step
# 4의 예측값은 :  [[4.0000014]]