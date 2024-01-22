import numpy as np
from keras.datasets import mnist
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)   # (10000, 28, 28) (10000,)

print(x_train)
print(x_train[0])
print(y_train[0])   # 5
print(np.unique(y_train, return_counts=True)) 
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
print(pd.value_counts(y_test))
# 1    1135
# 2    1032
# 7    1028
# 3    1010
# 9    1009
# 4     982
# 0     980
# 8     974
# 6     958
# 5     892

x_train = x_train.reshape(60000, 28, 28, 1)
# x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# print(x_train.shape[0]) # 60000
print(x_train.shape, x_test.shape)

# 2. 모델
model = Sequential()
model.add(Conv2D(8, (2,2), strides=1,
                 padding='same',
                #  padding='valid',
                
                 input_shape=(10, 10, 1),
                 
                 ))    # 9 투입하는거 명칭 : filter    / 아래 레이어로 이동할때 (N,27,27,9) 로 변경
model.add(MaxPooling2D())   # 커널사이즈로 대입하였을때 가장 큰값을 추출
                            # (2,2) 일때는 n / 2 , (3,3) 일때는 n / 3 
model.add(Conv2D(filters=7, kernel_size=(2,2)))    # 아래 레이어로 이동할때 (N, 25, 25, 10)
model.add(Conv2D(15, (4,4)))    # 아래 레이어로 이동할때 (N, 22, 22, 15)
# shape=(batch_size, rows, coulumns, channels)
# shape=(batch_size, height, width, channels)
model.add(Flatten())    # (N, 22, 22, 15) 가져와서 작업
model.add(Dense(8))     # (N, 27, 27, 8)
model.add(Dense(7, input_shape=(8,)))     # (N, 27, 27, 7)
model.add(Dense(6))     # (N, 27, 27, 6)
model.add(Dense(10, activation='softmax'))  # (N, 27, 27, 10)

model.summary()


'''
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])   # 다중분류 - 카테고리컬
model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=100, validation_split=0.2)

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print('loss', results[0])
# print('acc', results[1])

print(results)




#     ValueError: Shapes (32,) and (32, 27, 27, 10) are incompatible

'''
'''
import matplotlib.pyplot as plt
plt.imshow(x_train[8888], 'gray')
plt.show()
'''
'''
'''