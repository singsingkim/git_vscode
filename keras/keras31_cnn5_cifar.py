from keras.datasets import cifar10
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import time

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)
print(np.unique(y_train, return_counts=True))
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],

# 맹글기    # acc = 0.77 이상

print(x_train.shape, x_test.shape)  # (50000, 32, 32, 3) (10000, 32, 32, 3)

ohe = OneHotEncoder(sparse=False)   # sparse 디폴트값은 true 이며 matrix 를 반환한다.
                                    # sparse = False 를 주어야 array 를 반환한다
y_train=y_train.reshape(-1, 1)
y_test=y_test.reshape(-1, 1)
y_train=ohe.fit_transform(y_train)
y_test=ohe.fit_transform(y_test)

x_train,x_test,y_train,y_test=train_test_split(
    x_train, y_train, train_size=0.8,
    random_state=112, stratify=y_train)

# 2 모델
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['acc'])   # 다중분류 - 카테고리컬
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=6000,
                verbose=1,
                restore_best_weights=True
                )

start_time = time.time()
model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=10000,
          validation_split=0.2, callbacks=[es])
end_time = time.time()

# 4. 평가, 예측
results = model.evaluate(x_test, y_test)
# print('loss', results[0])
# print('acc', results[1])

print("loss : ", results[0])
print("acc : ", results[1])

import matplotlib.pyplot as plt
plt.imshow(x_train[200], 'gray')
plt.show()

# 맹글기    # acc = 0.77 이상

# loss :  2.1754391193389893
# acc :  0.33250001072883606