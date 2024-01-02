import numpy as np
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import keras

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]
             )

y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape)
print(y.shape)

x = x.T
print(x.shape)

model = Sequential()
model.add(Dense(1, input_dim=2))
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x,y,epochs=1000, batch_size=2)

loss = model.evaluate(x,y)
result = model.predict([[10,1.3]])
print("[10, 1.3] 의 예측값 : ", result)

