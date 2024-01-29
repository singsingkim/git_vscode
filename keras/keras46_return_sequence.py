import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM 
from keras.callbacks import EarlyStopping

# 1 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_pred = np.array([50,60,70]).reshape(1,3,1)

print(x.shape, y.shape) # (13, 3) (13,)
    # 3차원 모양 맞춰주기 위해 리쉐이프 필요

x = x.reshape(13,3,1)
print(x.shape, y.shape) # (13, 3, 1) (13,)

# 2 모델
model = Sequential()
# model.add(SimpleRNN(units=512, input_shape=(3,1))) # (timesteps, features)
# 3-D tensor with shape (batch_size, timesteps, features).
model.add(LSTM(units=256, return_sequences=True,
               input_shape=(3,1))) # (timesteps, features)
# model.add(SimpleRNN(units=256, input_shape=(3,1))) # (timesteps, features)
model.add(LSTM(128, return_sequences=True))
model.add(Dense(32))
model.add(LSTM(128))
# model.add(LSTM(256))
model.add(Dense(32))
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
y_pred = model.predict(x_pred)
# (3,) -> (1, 3, 1)
print('로스는 : ',results)
print('[50,60,70] 의 결과 : ', y_pred)

# 에포 500
# 로스는 :  0.03482223302125931
# 1/1 [==============================] - 1s 556ms/step
# [50,60,70] 의 결과 :  [[77.99921]]

# 에포 1000
# 로스는 :  0.0002776515029836446
# [50,60,70] 의 결과 :  [[78.9481]]