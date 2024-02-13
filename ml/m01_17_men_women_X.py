#Train Test 를 분리해서 해보기
#불러오는데 걸리는 시간.
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
from sklearn.svm import LinearSVC

# 1 데이터
image_path = 'c:/_data/image/man_and_women/test//'
path = 'c:/_data/kaggle/man_women//'
np_path = 'c:/_data/_save_npy/'

x_train = np.load(np_path + 'keras39_5_x_train.npy')
y_train = np.load(np_path + 'keras39_5_y_train.npy')
x_test = np.load(np_path + 'keras39_5_x_test.npy')
y_test = np.load(np_path + 'keras39_5_y_test.npy')

print(x_train.shape, y_train.shape) # (3309, 200, 200, 3) (3309,)
print(x_test.shape, y_test.shape)   # (3309, 200, 200, 3) (3309,)

#증폭
data_generator =  ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=20,
    zoom_range=0.2
)

augument_size = 1700

#rand
randidx = np.random.randint(x_train.shape[0], size= augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 3)

x_augumented = data_generator.flow(
    x_augumented, y_augumented,
    batch_size=augument_size,
    shuffle=False
).next()[0]

#reshape
x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 3)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)

#concatenate
x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape)
print(x_test.shape)

#scailing
x_train = x_train/255.
x_test = x_test/255.

# #onehot
# ohe = OneHotEncoder(sparse=False)
# y_train =  ohe.fit_transform(y_train.reshape(-1,1))
# y_test =  ohe.fit_transform(y_test.reshape(-1,1))

print(x_train.shape, y_train.shape) # (5009, 200, 200, 3) (5009,)
print(x_test.shape, y_test.shape)   # (3309, 200, 200, 3) (3309,)


#2 모델구성
model = LinearSVC(C = 100)
# model = Sequential()
# model.add(Conv2D(64, (2,2), input_shape=(200,200,3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.1))
# model.add(Conv2D(128, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.2))
# model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid'))

# from keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', mode = 'min', patience=100, restore_best_weights=True)

#3 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(x_train, y_train, epochs= 1, batch_size= 50, validation_split= 0.2, callbacks=[es])

#4 평가 예측
results = model.score(x_test, y_test)
print('results : ', results)
# loss = model.evaluate(x_test, y_test)
# predict = np.round(model.predict(x_test))#.flatten()

# print('loss : ', loss[0])
# print('acc : ', loss[1])
# print(predict)
# print(len(predict))



# '''
# loss :  0.46612799167633057
# acc :  0.7822499871253967

# loss :  0.4787946939468384
# acc :  0.7768844366073608
# '''