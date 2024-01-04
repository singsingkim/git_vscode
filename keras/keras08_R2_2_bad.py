# 고의적으로 R2값 낮추기.
# 1. R2를 음수가 아닌 0.5 미만으로 만들것
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋과 아웃풋 포함해서 7개 이상
# 4. batch_size = 1
# 5. 히든레이어의 노드는 10개 이상 100개 이하
# 6. train 사이즈 75%
# 7. epochs 100번 이상
# [실습 시작]



import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    shuffle=True,
                                                    random_state=39)    # 랜덤 난수. 성능 향상을 노릴수 있음
# 모델 구성후 그려보자

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(13))
model.add(Dense(16))
model.add(Dense(13))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

loss = model.evaluate(x_test, y_test)
y_predict = model.predict([x_test])
result = model.predict(x)

print("로스 : ", loss)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print("R2 스코어 : ", r2)

import matplotlib.pyplot as plt

# ★ 시각화 ★
# plt.scatter(x,y)
# plt.plot(x, result, color='red')    # plot 을 scatter 로 바꾸면 점으로 실제 데이터가 직선으로 찍힘
# plt.show()


# Epoch 100/100
# 15/15 [==============================] - 0s 125us/step - loss: 11.2417
# 1/1 [==============================] - 0s 150ms/step - loss: 13.1775
# 1/1 [==============================] - 0s 49ms/step
# 1/1 [==============================] - 0s 50ms/step
# 로스 :  13.177515029907227
# R2 스코어 :  0.2775485707174704