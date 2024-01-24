# 캣독 데이터 저장
# x, y 추출해서 모델 맹글기
# 성능 0.99 이상

# 이미지를 수치화
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
x_train = np.load(np_path + 'keras39_3_x_train.npy')
y_train = np.load(np_path + 'keras39_3_y_train.npy')
x_test = np.load(np_path + 'keras39_3_x_test.npy')
y_test = np.load(np_path + 'keras39_3_y_test.npy')

print(x_train.shape, y_train.shape) # (19995, 100, 100, 1) (19995,)
print(x_test.shape, y_test.shape)   # (4998, 100, 100, 1) (4998,)

x_train, x_test, y_train, y_test = train_test_split(
    x_train, y_train, 
    test_size=0.2, 
    random_state=4756, 
    stratify=y_train)


# 2 모델
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape = (100, 100, 1),
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
model.add(Dense(1, activation='sigmoid'))

model.summary()

# 3 컴파일, 훈련
model.compile(loss= 'binary_crossentropy', 
              optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=4000,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, batch_size=32, 
          verbose= 1, epochs= 1, validation_split=0.2,
          callbacks=[es] 
            )

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time,2),"초")

# 4.평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print(y_predict)
# [[0.5420027 ]
#  [0.5273362 ]
#  [0.5502516 ]
#  ...
#  [0.52849936]
#  [0.5078639 ]
#  [0.3964163 ]]
y_predict = np.around(y_predict)
print(y_predict)







# print('loss = ', results[0])
# print('acc = ', results[1])



