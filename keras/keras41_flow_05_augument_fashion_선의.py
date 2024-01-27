from keras.datasets import fashion_mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
from keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)

#증폭
data_generator =  ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=20,
    zoom_range=0.2
)

augument_size = 40000

#rand
randidx = np.random.randint(x_train.shape[0], size= augument_size)

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy()

x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = data_generator.flow(
    x_augumented, y_augumented,
    batch_size=augument_size,
    shuffle=False
).next()[0]

#reshape
x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

#concatenate
x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape)
print(x_test.shape)

#scailing
x_train = x_train/255.
x_test = x_test/255.

#onehot
ohe = OneHotEncoder(sparse=False)
y_train =  ohe.fit_transform(y_train.reshape(-1,1))
y_test =  ohe.fit_transform(y_test.reshape(-1,1))



#2.모델구성
model = Sequential()
model.add(Conv2D(64, (2,2), input_shape = (28,28,1), activation='relu'))
model.add(Dropout(0.15))

model.add(Conv2D(128, (2,2), activation='relu'))
model.add(Dropout(0.3))

model.add(Conv2D(256, (2,2), activation='relu'))
model.add(Dropout(0.5))

model.add(Flatten())

#certificate
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=1, batch_size= 400, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=30, restore_best_weights=True)
])

#4.모델 평가, 예측

predict = ohe.inverse_transform(model.predict(x_test))
print(predict)
y_test = ohe.inverse_transform(y_test)
acc_score = accuracy_score(y_test, predict)
print('acc_score :', acc_score)

'''
===============   증폭 전     =================
acc_score : 0.9195
===============40000개 증폭 후=================
acc_score : 0.9206
'''