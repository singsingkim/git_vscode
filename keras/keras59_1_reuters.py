from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, 
                                                         test_split=0.2)

print(x_train)
print(x_train.shape, y_train.shape) # (8982,) (8982,)
print(x_test.shape, y_test.shape)   # (2246,) (2246,)
print(y_train)                  # [ 3  4  3 ... 25  3 25]
print(len(np.unique(y_train)))  # 46
print(len(np.unique(y_test)))  # 46

print(type(x_train))    # <class 'numpy.ndarray'>
print(type(x_train[0])) # <class 'list'>
print(len(x_train[0]), len(x_train[1])) # 87 56

print('뉴스기사의 최대길이 :', max(len(i) for i in x_train))
# x_train 의 가장 최대 값 : 2376
print('뉴스기사의 평균길이 :', sum(map(len, x_train))/len(x_train)) 
# 145.5398574927633

# 전처리
from keras.utils import pad_sequences
x_train = pad_sequences(x_train, padding='pre', maxlen=100, truncating='pre')
x_test = pad_sequences(x_test, padding='pre', maxlen=100, truncating='pre')
# y원핫은 하고싶으면 하고  하기싫으면 sparse+categorical_crossentropy

print(x_train.shape, x_test.shape)  # (8982, 100) (2246, 100)

# 맹글맹글
# 임배디드 모델 파라미터 단어사전갯수, 아웃풋딤 임의, 인풋랭스 = max(len) 같다


# 모델 생성
model = Sequential()

# # 임베딩 레이어 추가
# vocab_size = 1000  # 단어 사전의 크기
# embedding_dim = 100  # 임베딩 차원
# maxlen = 100  # 입력 시퀀스의 최대 길이

model.add(Embedding(10000, 100))
model.add(LSTM(64))  # LSTM의 유닛 수는 임의로 선택 가능
model.add(Dense(46, activation='softmax'))

model.summary()

# 3 컴파일 , 훈련
es = EarlyStopping(monitor='val_loss', # 발리데이션 스플릿과 세트이다
                   mode='min',
                   patience=500,
                   verbose=1,
                   restore_best_weights=True)

model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])

model.fit(x_train, y_train,
          epochs=1000, verbose=1,
          callbacks=[es], batch_size=128, validation_split=0.2)

# 4 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print('로스 : ', results[0])
print('ACC : ', results[1])


# 로스 :  1.6384590864181519
# ACC :  0.6108637452125549              
              
              