from keras.models import Sequential
from keras.layers import Dense
import keras
import tensorflow as tf
import numpy as np
import random as rn

print(keras.__version__)    # 2.9.0
print(tf.__version__)       # 2.9.0
print(np.__version__)       # 1.26.3
rn.seed(333)
tf.random.set_seed(123)     # 텐서 2.9 에서는 고정 가능. 2.12 고정 불가
np.random.seed(321)


# 1 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2 모델
model = Sequential()
model.add(Dense(5, 
                # kernel_initializer='zeros',
                input_dim=1))
model.add(Dense(5))
model.add(Dense(1))

# 3
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, verbose=0)

# 4
loss = model.evaluate(x, y, verbose=0)
results = model.predict([4], verbose=0)
print('loss : ', loss)
print('4의 예측값 : ', results)