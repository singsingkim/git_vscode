# x, y 추출해서 모델 맹글기
# 성능 0.99 이상
'''
x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
'''

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
# 증폭을 정의한것 -> 실행은 안시킴
train_datagen = ImageDataGenerator( # 컴퓨터가 알아먹을수 있게 숫자로 변환
                                   # 약 20000 장의 데이터 수집 -> 각각 크기가 다르다 -> 규격다르고 많기때문에 오래걸린다
        # 증폭 과정                     # mnist 는 이미 숫자 규격이어서 편했다
    rescale=1./255,         # . 점이 있다는거는 부동소수점으로 연산 한다는거
    # horizontal_flip=True,   # 수평 뒤집기
    # vertical_flip=True,     # 수직 뒤집기
    # width_shift_range=0.1,  # 평행 이동
    # height_shift_range=0.1, # 
    # rotation_range=5,       # 정해진 각도만큼 이미지를 회전
    # zoom_range=1.2,         # 축소 또는 확대
    # shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    # fill_mode='nearest',    # 이동시켰을때 빈공간이 생기면 근처 값의 비슷한값으로 빈공간을 채운다
        # 우선 우리가 원하는건 수치화이기 때문에 수치화 과정에서는 증폭 생략 가능 (주석처리)
        # 주석처리해서 돌리는 이유는 코딩이 완벽하지 않은 상태에서 증폭과정을 넣어주면 확인처리가 너무 오래걸린다
        # 그렇기 때문에 증폭과정을 생략하고 우선 코딩이 완벽한지 확인후 증폭과정을 넣어준다
)

test_datagen = ImageDataGenerator(     # 테스트 데이터는 트레인 데이터를 테스트에 맞춰보아야 하기 때문에 증폭을 시키면 데이터손실로 이어진다
    rescale=1./255,
    
)

path_train = 'c:/_data/image/brain/train//'
path_test = 'c:/_data/image/brain/test//'

# 실제 실행시키는 부분
xy_train = train_datagen.flow_from_directory(   # xy 가 한번에 있다. 사이킷런 데이터에서 데이터와 타겟으로 나눈거와 비슷
    # 이터레이터 : 반복자
    path_train, 
    target_size=(100, 100),     # 사이즈 조정
    batch_size=200,           # 몇장씩 수치화를 할지/ 20000 이면 한 번 돌면서 전부 수치화
                                # 기존 배치사이즈와 동일/ 200,200 사이즈로 5개씩
                                # 7억 인구를 수치화 한다면 배치사이즈 조절후에 이어붙여야한다/ 업엔드? 컹그랫?
                                # 
    # 배치사이즈 160 이상을 쓰면 x 통데이터로 가져올수 있다
    # 통데이터 받고싶어서 배치사이즈 160, 배치에 더욱 많이 넣어도 160 나온다.
    
    class_mode='binary',
    color_mode='grayscale',     # 컬러는? 'rpg'
    shuffle=True
        
)   # Found 20000 images belonging to 2 classes.

# print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x00000278AA9F2D30>
# 데이터는 프리프로패싱에 감싸져 있다. 쉐이프를 빼내거나

# xy_ test
xy_test = test_datagen.flow_from_directory(   # xy 가 한번에 있다. 사이킷런 데이터에서 데이터와 타겟으로 나눈거와 비슷
    # 이터레이터 : 반복자
    path_test, 
    target_size=(100, 100),         # 사이즈 조정
    batch_size=200,                # 기존 배치사이즈와 동일/ 200,200 사이즈로 5개씩
    class_mode='binary',
    color_mode='grayscale',     # 컬러는? 'rpg'
        
)   

print(xy_train[0][0])   # 첫번째 배치의 x
print(xy_train[0][1])   # [0. 1. 0. ... 0. 0. 0.]   # 첫번째 배치의 y

print(xy_train[0][0].shape) # (160, 100, 100, 1)
print(xy_train[0][1].shape) # (160, )

print(xy_test[0][0].shape) # (120, 100, 100, 1)
print(xy_test[0][1].shape) # (120,)

np_path = 'c:/_data/_save_npy//'
np.save(np_path + 'keras39_1_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras39_1_y_train.npy', arr=xy_train[0][1])
np.save(np_path + 'keras39_1_x_test.npy', arr=xy_test[0][0])
np.save(np_path + 'keras39_1_y_test.npy', arr=xy_test[0][1])



#=============================================================
# one_hot = OneHotEncoder()
# y_train = one_hot.fit_transform(y_train.reshape(-1, 1)).toarray()
# y_test = one_hot.transform(y_test.reshape(-1, 1)).toarray()


# y_train=y_train.reshape(-1,1) # 다중분류일때 사용
# y_test=y_test.reshape(-1,1)

# print(x_train.shape, y_train.shape) # (14996, 100, 100, 3) (14996,)
# print(x_test.shape, y_test.shape)   # (4999, 100, 100, 3) (4999,)
# print(y_train)  # [0. 0. 0. ... 0. 1. 0.]
# print(pd.value_counts(y_test))
# # 1.0    2500
# # 0.0    2499
# print(np.unique(y_train, return_counts=True)) 
# (array([0., 1.], dtype=float32), array([7497, 7499], dtype=int64))





'''
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

# input1 = Input(shape=(200*200*3,))
# dense1 = Dense(100, activation='relu')(input1)
# drop1 = Dropout(0.3)(dense1)
# dense2 = Dense(10, activation='relu')(drop1)
# drop2 = Dropout(0.3)(dense2)
# dense3 = Dense(100, activation='relu')(drop2)
# drop3 = Dropout(0.3)(dense3)
# dense4 = Dense(100, activation='relu')(drop3)
# dense5 = Dense(60, activation= 'relu')(dense4)
# drop4 = Dropout(0.3)(dense5)
# dense6 = Dense(30, activation= 'relu')(drop4)
# output1 = Dense(1, activation= 'sigmoid')(dense6)
# model = Model(inputs=input1, outputs=output1)

model.summary()

# print(x_train.shape, y_train.shape)


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
model.fit(xy_train,   # 핏제너레이터 -> 통으로 던져주면 알아서 분배 
          # batch_size=32,      # fit_generator 에서는 에러, fit 에서는 안먹힘                            # 배치사이즈 설정은 하고싶으면 위에서 해야한다
          steps_per_epoch=16,   # # 전체데이터 / batch = 160/10 = 16
          # 17 은 에러, 15 는 데이터 손실
          verbose= 1,
          epochs= 10,           
          validation_data=xy_test,
          #   validation_split=0.2    # 에러
          callbacks=[es] 
            )

end_time = time.time()
print("걸린 시간 : ", round(end_time - start_time,2),"초")

# 4.평가, 예측
results = model.evaluate(xy_test)
print('loss = ', results[0])
print('acc = ', results[1])

'''