from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])
# 위 데이터를 훈련해서 최소의 loss를 만들자

#2. 모델구성 y = wx + b
model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([1,2,3,4,5,6,7])
print("7의 예측값 : ", result)



# 로스값이 변하지 않는 구간부터 과적합 구간

# Epoch 10000/10000 
# 1/1 [==============================] - 0s 0s/step - loss: 0.3238
# 1/1 [==============================] - 0s 50ms/step - loss: 0.3238
# 로스 :  0.3238094449043274
# 1/1 [==============================] - 0s 38ms/step
# 7의 예측값 :  [[6.8]]


# Epoch 10000/10000
# 1/1 [==============================] - 0s 0s/step - loss: 0.3238
# 1/1 [==============================] - 0s 60ms/step - loss: 0.3238
# 로스 :  0.32380956411361694
# 1/1 [==============================] - 0s 39ms/step
# 7의 예측값 :  [[1.1428591]
#  [2.0857155]
#  [3.0285718]
#  [3.9714282]
#  [4.9142847]
#  [5.857141 ]
#  [6.7999973]]