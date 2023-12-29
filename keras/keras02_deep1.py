from keras.models import Sequential
from keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(3, input_dim=1))    # 노드의 갯수와 레이어의 깊이를 조절 가능 
model.add(Dense(10, input_dim=3))    
model.add(Dense(10))
model.add(Dense(30))
model.add(Dense(50))    # 레이어를 추가하는 것
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(110))
model.add(Dense(105))
model.add(Dense(100))
model.add(Dense(90))
model.add(Dense(80))
model.add(Dense(70))
model.add(Dense(50))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))     # 인풋딤 생략 가능
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(1))




#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100)

#4. 평가, 예측
loss = model.evaluate(x, y)
print("로스는 : ", loss)
result = model.predict([4])
print("4의 예측값 : ", result)



# Epoch 100/100
# 1/1 [==============================] - 0s 2ms/step - loss: 3.2637e-05
# 1/1 [==============================] - 0s 98ms/step - loss: 9.4808e-06
# 로스는 :  9.480791959504131e-06
# 1/1 [==============================] - 0s 91ms/step
# 4의 예측값 :  [[3.9993021]]


# Epoch 100/100
# 1/1 [==============================] - 0s 391us/step - loss: 5.9891e-06
# 1/1 [==============================] - 0s 100ms/step - loss: 2.5269e-06
# 로스는 :  2.5268589070037706e-06
# 1/1 [==============================] - 0s 115ms/step
# 4의 예측값 :  [[4.000899]]


# Epoch 100/100
# 1/1 [==============================] - 0s 2ms/step - loss: 5.7887e-05
# 1/1 [==============================] - 0s 105ms/step - loss: 9.4523e-05
# 로스는 :  9.452344966121018e-05
# 1/1 [==============================] - 0s 108ms/step
# 4의 예측값 :  [[4.002781  3.9999058]]


# Epoch 100/100
# 1/1 [==============================] - 0s 0s/step - loss: 6.2478e-05
# 1/1 [==============================] - 0s 117ms/step - loss: 1.9137e-06
# 로스는 :  1.9136659830110148e-06
# 1/1 [==============================] - 0s 113ms/step
# 4의 예측값 :  [[3.999935]]