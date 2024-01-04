import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=400)    # 랜덤 난수. 성능 향상을 노릴수 있음
# 모델 구성후 그려보자

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=2)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict([x_test])
result = model.predict(x)

print("로스 : ", loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 : ", r2)

import matplotlib.pyplot as plt

# ★ 시각화 ★
plt.scatter(x,y)
plt.plot(x, result, color='red')    # plot 을 scatter 로 바꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.show()

# Epoch 1000/1000
# 7/7 [==============================] - 0s 0s/step - loss: 14.8958
# 1/1 [==============================] - 0s 77ms/step - loss: 4.3880
# 1/1 [==============================] - 0s 72ms/step
# 1/1 [==============================] - 0s 33ms/step
# 로스 :  4.387991428375244
# R2 스코어 :  0.919935287095894