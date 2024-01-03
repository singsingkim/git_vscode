import numpy as np
from keras.models import Sequential
from keras.layers import Dense

x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,  1.5, 1.4, 1.3],
              [9,8,7,6,5,4,3,2,1,0]
            ])
y = np.array([1,2,3,4,5,6,7,8,9,10])

print(x.shape, y.shape) # (3, 10) (10, )
x = x.T
print(x.shape) # (10, 3)

model = Sequential()
model.add(Dense(1, input_dim = 3))    # n행 3열  # 디맨션이 3개란 의미. 아래 #4 의 예측값도 열의 형태로 변해주어야함
# 열, 컬럼, 속성, 특성, 차원 = 3 // 같다
# (행무시, 열우선) <= 암기
model.add(Dense(1))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam')
model.fit(x, y, epochs = 1000, batch_size = 2)

# 4. 평가, 예측
loss = model.evaluate(x, y)
result = model.predict([[10, 1.3, 0]])      # predict([10, 1.3, 0]) --> 형태가 (3, ) 인 스칼라 0차원 형태이다.
                                            # predict([[10, 1.3, 0]]) --> 형태를 (1, 3) 1행 3열 벡터 형태인 1차원으로 만들어 주어야한다.
                                            # (100, 3) 이나 (200, 3) 등등 (N, 3) N 콤마 3 열 형태로 인식해야 한다.

print("[10, 1.3, 0]의 예측값 : ", result)
# [실습] : 소수 2째 자리 까지 맞추기

# Epoch 1000/1000
# 5/5 [==============================] - 0s 4ms/step - loss: 6.0125e-08
# 1/1 [==============================] - 0s 70ms/step - loss: 1.2015e-07
# 1/1 [==============================] - 0s 56ms/step
# [10, 1.3, 0]의 예측값 :  [[10.0004015]]


# 프레딕을 10, 1.3, 0 을 주는 이유는 x 1.3 다음에 무엇이 올 지 모르기 때문에 
# 이미 훈련을 완료한 10, 1.3, 0 을 프레딕에 넣어 답이 나와 있는 10에 가깝게 예측 하는지 확인 