import numpy as np
from keras.models import Sequential
from keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,6,5,7,8,9,10])

# [검색] train과 test를 섞어서 7:3 으로 랜덤하게 분할 방법 찾기
# 힌트 : 사이킷런


from sklearn.model_selection import train_test_split
#학습용 데이터와 테스트 데이터로 분리
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,       # 디폴트 : 0.75
                                                    test_size=0.3, 
                                                    shuffle=True,         # 디폴트 True
                                                    random_state=1)

print(x_train)
print(y_train)
print(x_test)
print(y_test)

model = Sequential()
model.add(Dense(10 , input_dim = 1))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 1000, batch_size = 2)

loss = model.evaluate(x_test, y_test)
results = model.predict([x])

# print("로스 : ", loss)
# print("x_train", x_train)
# print("y_train", y_train)
# print("x_test", x_test)
# print("y_test", y_test)

print("로스 : ", loss)
print("예측값 : ", results)

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.plot(x, results, color='red')
plt.show()