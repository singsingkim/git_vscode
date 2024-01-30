import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Bidirectional
from keras.callbacks import EarlyStopping

# 1 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape, y.shape) # (13, 3) (13,)
    # 3차원 모양 맞춰주기 위해 리쉐이프 필요

x = x.reshape(13,3,1)
print(x.shape, y.shape) # (13, 3, 1) (13,)

# 2 모델
model = Sequential()
# model.add(SimpleRNN(units=512, input_shape=(3,1))) # (timesteps, features)
# 3-D tensor with shape (batch_size, timesteps, features).
model.add(Bidirectional(LSTM(units=256, ),input_shape=(3,1))) # (timesteps, features)
# model.add(Bidirectional(LSTM(256),input_shape=(3,1))) # (timesteps, features)
# model.add(SimpleRNN(units=256, input_shape=(3,1))) # (timesteps, features)
model.add(Dense(32))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,))

model.summary()

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x, y, epochs=1000, callbacks=[es],
          batch_size=64)

# 4 평가, 예측
results  = model.evaluate(x,y)
print('로스는 : ',results)
y_pred = np.array([50,60,70]).reshape(1,3,1)
y_pred = model.predict(y_pred)
# (3,) -> (1, 3, 1)
print('[50,60,70] 의 결과 : ', y_pred)

# 에포 5000
# 로스는 :  5.171003067516722e-05
# 1/1 [==============================] - 0s 262ms/step
# [50,60,70] 의 결과 :  [[78.87817]]

# 에포 1000
# 로스는 :  0.00019505379896145314
# 1/1 [==============================] - 0s 267ms/step
# [50,60,70] 의 결과 :  [[79.6833]]

# 바이디랙셔널 에포 1000
# 로스는 :  0.00014651463425252587
# 1/1 [==============================] - 0s 293ms/step
# [50,60,70] 의 결과 :  [[79.6362]]