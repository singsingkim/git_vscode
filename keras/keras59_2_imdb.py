from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
from collections import Counter

# 1 데이터
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)

print(x_train)
print(x_train.shape, y_train.shape) # (25000,) (25000,)
print(x_test.shape, y_test.shape)   # (25000,) (25000,)
print(type(x_train))                 # <class 'numpy.ndarray'>
print(type(x_train[0]))                 # <class 'list'>
print(len(x_train[0]), len(x_test[0]))  # 218 68
print(y_train[:20]) # [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
counter = Counter(y_train)
print(counter)
# Counter({1: 12500, 0: 12500})
# y 값이 0 과 1 두 개 뿐. -> 이진분류


print('최대길이 : ', max(len(i) for i in x_train))          # 2494
print('평균길이 : ', sum(map(len, x_train))/len(x_train))   # 238.71364
# 패딩시퀀스에서  맥스렌을 최대길이가 아닌 평균길이에 맞추는 이유
# 최대길이에 맞추게 되면 쓸데없는 패딩이 많이 생기게 되므로 문장에서 중요부는 보통
# 말미에 있으니 패딩을 앞에서부터 채우고 커팅도 앞에서부터 커팅한다

# 전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=250, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=250, truncating='pre')
print(x_train.shape)    # (25000, 250)
print(x_test.shape)     # (25000, 250)


# 2 모델
model = Sequential()
model.add(Embedding(10000, 77))
model.add(LSTM(128))
model.add(Dense(16))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1, activation='sigmoid'))

model.summary()


# 3 컴파일, 훈련
es = EarlyStopping(monitor='loss',
                   mode='min',
                   patience=500,
                   verbose=1,
                   restore_best_weights=True)

model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train, epochs=10, batch_size=2048, callbacks=[es])

# 4 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
y_argpre = np.argmax(y_predict, axis=1)
y_aroundpre = np.around(y_predict)
print(y_predict[:5])
print(y_argpre[:5])
print(y_aroundpre[:5])
f1 = f1_score(y_test, y_aroundpre, average='macro')
print('로스 : ', loss[0])
print('ACC : ', loss[1])
print('F1 : ', f1)


