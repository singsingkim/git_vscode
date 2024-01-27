from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

# 1 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 스케일링
x_train = x_train/255.
x_test = x_test/255.

# 데이터 형태 증폭
train_datagen = ImageDataGenerator(
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
randidx = np.random.randint(x_train.shape[0], size=augumet_size)
# 같다  # np.random.randint(60000, 40000)   60000 만개 중에 40000만 개를 임의로 뽑아라

print(randidx)  # [56275 57046 16669 ...  7167 10671 46307]
print(np.min(randidx), np.max(randidx)) # 0 59999

x_augumented = x_train[randidx].copy()  # 주소가 공유되는걸 방지하기 위해 안전빵으로 원데이터에 영향을 미치지 않기 위해 .copy() 사용     # x 증폭
y_augumented = y_train[randidx].copy()  #  위와 같다
print(x_augumented)
print(x_augumented.shape)   # (40000, 28, 28)

print(y_augumented)
print(y_augumented.shape)   # (40000,)

# 리쉐이프
x_augumented = x_augumented.reshape(x_augumented.shape[0], x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(
    x_augumented, y_augumented, # 플로우에서 4차원이 들어가야하는데 아큐먼트는 3차원이라 위에서 리쉐이프 해주어야한다
    batch_size = augumet_size,
    shuffle=False,
        
).next()[0]    # 6만개의 데이터서 4만개를 가져와서 변환시켰다 = 증폭과 같다

print(x_augumented.shape)   # (40000, 28, 28, 1)

print(x_train.shape)        # (60000, 28, 28)
x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)

print(x_train.shape, x_augumented.shape)
x_train = np.concatenate((x_train, x_augumented))    # 사슬처럼 엮다
y_train = np.concatenate((y_train, y_augumented))    # 사슬처럼 엮다

print(x_train.shape, y_train.shape)


