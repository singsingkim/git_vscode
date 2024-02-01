from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import function_package as fp
from keras.datasets import reuters

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=1000,
                                                         test_split=0.2
                                                         )

print(x_train.shape, x_test.shape)  # (8982,) (2246,)
print(y_train.shape, y_test.shape)  # (8982,) (2246,)
print(type(x_train[0])) # <class 'list'>
print(np.unique(y_train, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64), 
#  array([  55,  432,   74, 3159, 1949,   17,   48,   16,  139,  101,  124,
#         390,   49,  172,   26,   20,  444,   39,   66,  549,  269,  100,
#          15,   41,   62,   92,   24,   15,   48,   19,   45,   39,   32,
#          11,   50,   10,   49,   19,   19,   24,   36,   30,   13,   21,
#          12,   18], dtype=int64))
print(np.unique(y_test, return_counts=True))
# (array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
#        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45], dtype=int64), 
#  array([ 12, 105,  20, 813, 474,   5,  14,   3,  38,  25,  30,  83,  13,
#         37,   2,   9,  99,  12,  20, 133,  70,  27,   7,  12,  19,  31,
#          8,   4,  10,   4,  12,  13,  10,   5,   7,   6,  11,   2,   3,
#          5,  10,   8,   3,   6,   5,   1], dtype=int64))
labels_num = max(len(np.unique(y_train)),len(np.unique(y_test)))
print(labels_num)

len_list = [len(i) for i in x_train] + [len(i) for i in x_test] # 모든 데이터의 길이 모아둔 리스트
len_list = pd.Series(len_list)
w_length = int(len_list.quantile(q=0.75))  # 문장의 길이의 제3 사분위수 (75% 지점)
# print(max(len(i) for i in x_train))
# print(max(len(i) for i in x_test))
print(w_length) # 180
x_train = pad_sequences(x_train, maxlen=w_length)
x_test = pad_sequences(x_test, maxlen=w_length)
print(x_train.shape, x_test.shape)  # (8982, 180) (2246, 180)

word_max = max([max(i) for i in x_train] + [max(i) for i in x_test]) # 단어 사전의 개수 세는 용
print(word_max) # 99

### ohe ###
# 확인결과 0부터 45까지 예쁘게 들어가 있으므로 to_categorical이 편하다
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# model
model = Sequential()
model.add(Embedding(word_max+1,512))
model.add(GRU(512, input_shape=(w_length,1)))
model.add(Dropout(0.05))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(labels_num, activation='softmax'))

# compile & fit
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_acc', mode='auto', patience=100, verbose=1)
hist = model.fit(x_train,y_train, epochs=4096, batch_size=256, validation_data=(x_test,y_test), verbose=2, callbacks=[es])

# evaluate
loss = model.evaluate(x_test,y_test)

print(f"loss: {loss[0]}\nACC:  {loss[1]}")

# loss: 3.3554139137268066
# ACC:  0.6647372841835022