from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6])
y = np.array([1,2,3,5,4,6])
# 위 데이터를 훈련해서 최소의 loss를 만들자

#2. 모델구성 y = wx + b
#### [실습] 100epoch 고정에 01_1번과 같은 결과를 도출
model = Sequential()
model.add(Dense(5, input_dim=1))
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



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스 : ", loss)
result = model.predict([7])
print("7의 예측값 : ", result)



# 로스값이 변하지 않는 구간부터 과적합 구간

# Epoch 100/100
# 1/1 [==============================] - 0s 0s/step - loss: 0.3240
# 1/1 [==============================] - 0s 115ms/step - loss: 0.3238
# 로스 :  0.32384827733039856
# 1/1 [==============================] - 0s 118ms/step
# 7의 예측값 :  [[6.790501]]