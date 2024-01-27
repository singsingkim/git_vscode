from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)#(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)#(10000, 28, 28) (10000,)

#데이터 증폭
data_generator =  ImageDataGenerator(
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=30,
    zoom_range=0.2,
    shear_range=0.7,
    fill_mode='nearest',
)

# 데이터 수량 증폭
augumet_size = 40000
randidx = np.random.randint(x_train.shape[0], size= augumet_size)
# 같다  # np.random.randint(60000, 40000)   60000 만개 중에 40000만 개를 임의로 뽑아라

print(randidx)  # [56275 57046 16669 ...  7167 10671 46307]
print(np.min(randidx), np.max(randidx)) # 0 59999

x_augumeted = x_train[randidx].copy()
y_augumeted = y_train[randidx].copy()

x_augumeted = x_augumeted.reshape(x_augumeted.shape[0], x_augumeted.shape[1], x_augumeted.shape[2], 1)

# print(x_augumented.shape)

x_augumeted = data_generator.flow(
    x_augumeted, y_augumeted,
    batch_size=augumet_size,
    shuffle=False
).next()[0]

#reshape
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)

#concatenate
x_train = np.concatenate((x_train, x_augumeted))
y_train = np.concatenate((y_train, y_augumeted))

print(x_train.shape, y_train.shape)#(100000, 28, 28, 1) (100000,)
print(x_train.shape, y_train.shape)#(100000, 28, 28, 1) (100000,)


#scaling
x_train = x_train/255.
x_test = x_test/255.

#클래스 확인
print(np.unique(y_train, return_counts=True)) #class count :9

#onehot처리
ohe = OneHotEncoder(sparse=False)
y_train = ohe.fit_transform(y_train.reshape(-1,1))
y_test = ohe.transform(y_test.reshape(-1,1))


#2.모델구성
model = Sequential()
model.add(Conv2D(9, (2,2), input_shape = (28, 28, 1))) 
model.add(Conv2D(16, (3,3), activation='relu')) #전달 (N,25,25,10)
model.add(MaxPooling2D()) 
model.add(Conv2D(32,(4,4))) #전달 (N,22,22,15)
model.add(Flatten()) #평탄화
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

#3.모델 컴파일, 훈련

model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs= 1000, batch_size=1000, verbose= 1, validation_split=0.2, callbacks=[
    EarlyStopping(monitor='val_loss', mode='min', patience=100, restore_best_weights=True)
] )

#4.평가 ,예측
y_test = ohe.inverse_transform(y_test) 
print(y_test)

predict = ohe.inverse_transform(model.predict(x_test)) 
acc_score = accuracy_score(y_test, predict)
print("acc_score :", acc_score)


# # ===============   증폭 전     =================
# acc =  0.9858999848365784
# # ===============40000개 증폭 후=================
# acc_score : 0.9886

