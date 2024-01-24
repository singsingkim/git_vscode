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
# np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])
# np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])
# np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])
# np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])

x_train = np.load(np_path + 'keras39_1_x_train.npy')
y_train = np.load(np_path + 'keras39_1_y_train.npy')
x_test = np.load(np_path + 'keras39_1_x_test.npy')
y_test = np.load(np_path + 'keras39_1_y_test.npy')
print(x_train.shape, y_train.shape) # (160, 100, 100, 1) (160,)
print(x_test.shape, y_test.shape)   # (120, 100, 100, 1) (120,)


#=============================================================
# one_hot = OneHotEncoder()
# y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
# y_test = one_hot.transform(y_test.reshape(-1, 1)).toarray()


# y_train=y_train.reshape(-1,1) # 다중분류일때 사용
# y_test=y_test.reshape(-1,1)


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
model.add(Dense(16, activation='relu'))
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

# model.fit_generator(xy_train,   # 핏제너레이터 -> 통으로 던져주면 알아서 분배 
model.fit(x_train, y_train,   # 핏제너레이터 -> 통으로 던져주면 알아서 분배 
          batch_size=32,      # fit_generator 에서는 에러, fit 에서는 안먹힘                            # 배치사이즈 설정은 하고싶으면 위에서 해야한다
        #   steps_per_epoch=16,   # # 전체데이터 / batch = 160/10 = 16
          verbose= 1,
          epochs= 100,           
        #   validation_data=xy_test,
          validation_split=0.2,    # 에러
          callbacks=[es] 
            )

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time,2),"초")

# 4.평가, 예측
results = model.evaluate(x_test, y_test)
print('loss = ', results[0])
print('acc = ', results[1])

# 에포 100
# loss =  0.09503171592950821
# acc =  0.9583333134651184
