# 이미지를 수치화
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# 증폭을 정의한것 -> 실행은 안시킴
train_datagen = ImageDataGenerator(
    rescale=1./255,         # . 점이 있다는거는 부동소수점으로 연산 한다는거
    horizontal_flip=True,   # 수평 뒤집기
    vertical_flip=True,     # 수직 뒤집기
    width_shift_range=0.1,  # 평행 이동
    height_shift_range=0.1, # 
    rotation_range=5,       # 정해진 각도만큼 이미지를 회전
    zoom_range=1.2,         # 축소 또는 확대
    shear_range=0.7,        # 좌표 하나를 고정시키고 다른 몇 개의 좌표를 이동시키는 변환
    fill_mode='nearest',    # 이동시켰을때 빈공간이 생기면 근처 값의 비슷한값으로 빈공간을 채운다
    
)

test_datagen = ImageDataGenerator(     # 테스트 데이터는 트레인 데이터를 테스트에 맞춰보아야 하기 때문에 증폭을 시키면 데이터손실로 이어진다
    rescale=1./255,
    
)

path_train = 'c:/_data/image/brain/train//'
path_test = 'c:/_data/image/brain/test//'

xy_train = train_datagen.flow_from_directory(   # xy 가 한번에 있다. 사이킷런 데이터에서 데이터와 타겟으로 나눈거와 비슷
    # 이터레이터 : 반복자
    path_train, 
    target_size=(200, 200),     # 사이즈 조정
    batch_size=160,               # 기존 배치사이즈와 동일/ 200,200 사이즈로 5개씩
    # 배치사이즈 160 이상을 쓰면 x 통데이터로 가져올수 있다
    # 통데이터 받고싶어서 배치사이즈 160, 배치에 더욱 많이 넣어도 160 나온다.
    class_mode='binary',
    shuffle=True
        
)   # Found 160 images belonging to 2 classes.

print(xy_train)
# <keras.preprocessing.image.DirectoryIterator object at 0x000001EB2EB13520>
# 데이터는 프리프로패싱에 감싸져 있다. 쉐이프를 빼내거나


xy_test = train_datagen.flow_from_directory(   # xy 가 한번에 있다. 사이킷런 데이터에서 데이터와 타겟으로 나눈거와 비슷
    # 이터레이터 : 반복자
    path_test, 
    target_size=(200, 200),     # 사이즈 조정
    batch_size=10,               # 기존 배치사이즈와 동일/ 200,200 사이즈로 5개씩
    class_mode='binary',
        
)   # Found 160 images belonging to 2 classes.
print(xy_test)
# Found 120 images belonging to 2 classes.

print(xy_train.next())

print(xy_train[0])  # 첫번재 배치값 -> 첫번째 x 값
# print(xy_train[16]) # 160개의 사진에 10개의 배치사이즈 해주어서 16개의 데이터가 생성. 17번째는 없다.
#  (10, 200, 200, 1? 3? 칼라는 아직 안배움)

print(xy_train[0][0])   # 첫번째 배치의 x
print(xy_train[0][1])   # 첫번째 배치의 y
print(xy_train[0][0].shape) # (10, 200, 200, 3) 칼라가 3인 이유 : 흑백도 칼라이다 오케이, 칼라도 흑백이다 는 엑스

print(type(xy_train))
# <class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0]))    # <class 'tuple'>, x 가 튜플 첫번째
print(type(xy_train[0][0])) # <class 'numpy.ndarray'>
print(type(xy_train[0][1])) # <class 'numpy.ndarray'>
