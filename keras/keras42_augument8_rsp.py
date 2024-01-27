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
x=np.load(np_path + 'keras39_9_x_train.npy')
y=np.load(np_path + 'keras39_9_y_train.npy')

print(x.shape, y.shape) # (2520, 150, 150, 3) (2520, 3)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=0.2, 
    random_state=4756, 
    stratify=y)

#증폭
data_generator =  ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=20,
    zoom_range=0.2
)

augument_size = 1000

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

print(x_train.shape)    # (3016, 150, 150, 3)
print(x_test.shape)     # (504, 150, 150, 3)

#scailing
x_train = x_train/255.
x_test = x_test/255.

#onehot
# ohe = OneHotEncoder(sparse=False)
# y_train =  ohe.fit_transform(y_train.reshape(-1,1))
# y_test =  ohe.fit_transform(y_test.reshape(-1,1))


print(x_train.shape, y_train.shape) # (3016, 150, 150, 3) (3016, 3)
print(x_test.shape, y_test.shape)   # (504, 150, 150, 3) (504, 3)

# 2 모델
model = Sequential()
model.add(Conv2D(128, (2,2), input_shape = (150, 150, 3),
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
model.add(Dense(3, activation='softmax'))

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
          verbose= 1, epochs= 100, validation_split=0.2,
          callbacks=[es] 
            )

# 평가, 예측
results = model.evaluate(x_test, y_test)

print(f"LOSS: {results[0]:.4}\nACC:{results[1]:.4f}")
print('loss = ', results[0])
print('acc = ', results[1])

# LOSS: 1.31e-07
# ACC:1.0000
# loss =  1.3103488072374603e-07
# acc =  1.0


'''
'''