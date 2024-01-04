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
                                                    random_state=50)    # 랜덤 난수. 성능 향상을 노릴수 있음
# 모델 구성후 그려보자

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=2)

loss = model.evaluate(x_test, y_test)
result = model.predict([x])

print("로스 : ", loss)
print("예측값 : ", result)

import matplotlib.pyplot as plt

print(x_train)
print(x_test)
print(y_train)
print(y_test)

# ★ 시각화 ★
plt.scatter(x,y)
plt.plot(x, result, color='red')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.show()
