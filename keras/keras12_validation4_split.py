from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pandas as pd

#1. 데이터
x = np.array(range(1, 17))
y = np.array(range(1, 17))

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle = True, train_size = 0.8125, random_state=1)

print(x_train, y_train)
print(x_test, y_test)

#.2 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 2, 
                                # validation_data = (x_val, y_val)  # validation_data 검증
            validation_split = 0.3,
            verbose=1)    
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
result = model.predict([11000, 7])    # x에 넣어주는 다음 예측값을 통해 다음 y값을 예측

print("로스 : ", loss)
print("예측값 : ", result)