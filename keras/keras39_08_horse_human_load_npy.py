import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import time
start_time = time.time()
# 1 데이터

np_path = 'c:/_data/_save_npy//'
x=np.load(np_path + 'keras39_7_x_train.npy')
y=np.load(np_path + 'keras39_7_y_train.npy')

print(x.shape, y.shape) # (1027, 300, 300, 3) (1027, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=4756, 
    stratify=y)

print(x_train.shape, y_train.shape) # (821, 300, 300, 3) (821, 2)
print(x_test.shape, y_test.shape)   # (206, 300, 300, 3) (206, 2)

# 2 모델
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape = (300, 300, 3),
                 strides=2, padding='same')) 
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2), activation='relu')) #전달 (N,25,25,10)
model.add(Conv2D(128,(2,2))) #전달 (N,22,22,15)
model.add(MaxPooling2D())
model.add(Flatten()) #평탄화
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.summary()


# 3 컴파일, 훈련
model.compile(loss= 'categorical_crossentropy', 
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=200,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, batch_size=32, 
          verbose= 1, epochs= 1000, validation_split=0.2,
          callbacks=[es] 
            )

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time,2),"초")

# 4.평가, 예측
results = model.evaluate(x_test, y_test)
# # =================================================
# y_predict = model.predict(x_test)
# print(y_predict)
# #  [0.34291953 0.65708053]
# #  [0.32793632 0.6720637 ]
# #  [0.3186366  0.68136346]
# #  [0.40605456 0.5939455 ]
# #  [0.34048408 0.6595159 ]
# y_predict = np.around(y_predict.flatten())
# # print(y_predict, y_predict.shape)
# # [0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1. 0. 1.
# #  0. 1. 0. 1. 0. 1. 0. 1.] (200,)
# # 점선 안에는 서브밋할 때 필요한것
# # ===================================================
print(f"LOSS: {results[0]:.4}\nACC:{results[1]:.4f}")
print('loss = ', results[0])
print('acc = ', results[1])

# LOSS: 0.0001464
# ACC:1.0000
# loss =  0.00014636667037848383
# acc =  1.0