#Train Test 를 분리해서 해보기
#불러오는데 걸리는 시간.

from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os
# datapath = 'C:/Workspace/AIKONG/_data/'
datapath = 'C:/_data/'
image_path = datapath + 'image/man_and_women/test/'
path = datapath + 'kaggle/man_and_women/'
np_path = datapath + '_save_npy/'

x_train = np.load(np_path + 'keras39_5_x_train.npy')
y_train = np.load(np_path + 'keras39_5_y_train.npy')
x_test = np.load(np_path + 'keras39_5_x_test.npy')
y_test = np.load(np_path + 'keras39_5_y_test.npy')


print(x_train.shape)
print(x_test.shape)

# x_train = x_train/255.
# x_test = x_test/255.

#모델구성
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
model = Sequential()

model.add(Conv2D(64, (2,2), input_shape=(200,200,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.2))

# model.add(Conv2D(256, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.4))

# model.add(Conv2D(512, (3,3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode = 'min', patience=100, restore_best_weights=True)

#컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size= 50, validation_split= 0.2, callbacks=[es])

#평가 예측
loss = model.evaluate(x_test, y_test)
predict = np.round(model.predict(x_test)).flatten()

print('loss : ', loss[0])
print('acc : ', loss[1])
print(predict)
print(len(predict))


'''
loss :  0.46612799167633057
acc :  0.7822499871253967

loss :  0.4787946939468384
acc :  0.7768844366073608
'''