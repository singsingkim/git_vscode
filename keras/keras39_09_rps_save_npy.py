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

path = 'c:/_data/rps//'

xy_train = train_datagen.flow_from_directory(
    path,
    target_size=(150,150),
    batch_size=3000,
    class_mode='categorical',
    shuffle=True
)

print(xy_train[0][0].shape) # (2520, 150, 150, 3)
print(xy_train[0][1].shape) # (2520, 3)

np_path = 'c:/_data/_save_npy//'
np.save(np_path + 'keras39_9_x_train.npy', arr=xy_train[0][0])
np.save(np_path + 'keras39_9_y_train.npy', arr=xy_train[0][1])
print('성공')



