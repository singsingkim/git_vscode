# from tensorflow.keras.models import Seqeuntial
# import tensorflow
# from keras.models import Seqeuntial 
# -->
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import tensorflow as tf
import keras
print("tf 버전 : ", tf.__version__)
print("keras 버전 : ", keras.__version__)


# 1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])

# 2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1)) # ( 아웃풋, 인풋 )
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(50))    
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(200))
model.add(Dense(300))
model.add(Dense(500))
model.add(Dense(300))
model.add(Dense(200))
model.add(Dense(100))     
model.add(Dense(70))
model.add(Dense(60))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=100, batch_size=3) # 배치사이즈 기본값은 32

# 4. 평가, 예측
loss = model.evaluate(x,y)
result = model.predict([7])
print("로스 : ", loss)
print("7의 예측값 : ", result)


# Epoch 100/100 , batch_size = 3
# 2/2 [==============================] - 0s 1ms/step - loss: 0.3250
# 1/1 [==============================] - 0s 115ms/step - loss: 0.3238
# 1/1 [==============================] - 0s 100ms/step
# 로스 :  0.3238331377506256
# 7의 예측값 :  [[6.8089013]]