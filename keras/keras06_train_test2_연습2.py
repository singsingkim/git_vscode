import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([1,2,3,4,5,6,7,8,9,10])
#x = np.array([[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]])
y = np.array([[10,20,30,40,50,60,70,80,90,100],[100,90,80,70,60,50,40,30,20,10]])
x_train=x[:3:]  
#y_train=y[:3 ]

print(x_train)
#print(y_train)
'''
x_test=x[7:10:1]
y_test=y[7:]
print(x_train, x_test)
print(y_train, y_test)
print(x_test)
print(y_test)

model = Sequential()
model.add(Dense(10, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 2)

loss = model.evaluate(x_test, y_test)
result = model.predict([11000, 7])

print("로스 : ", loss)
print("[10] 의 예측값 : ", result)

# Epoch 1000/1000
# 4/4 [==============================] - 0s 0s/step - loss: 2.6189e-13
# 1/1 [==============================] - 0s 79ms/step - loss: 1.8190e-12
# 1/1 [==============================] - 0s 69ms/step
# 로스 :  1.8189894035458565e-12
# [10] 의 예측값 :  [[1.1000000e+04]
#  [7.0000005e+00]]

'''