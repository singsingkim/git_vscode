from tensorflow.python.keras.models  import Sequential
# from tensorflow.keras.models import Sequential
# from keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

model = Sequential()
# model.add(Dense(10, input_shape=(3,)))  # 인풋은 (n, 3)
model.add(Conv2D(10, (2,2), input_shape=(10,10,1))) # 흑백 1 / 컬러 3 / 다음 레이어로 전달해주는 레이어 값 10 / 2 x 2 짜른다 / 10 x 10 사이즈 이미지
model.add(Dense(5))
model.add(Dense(1))