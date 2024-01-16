from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
import time


# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image

# 설치하고 싶은 pip 가 있으면 pip install scikit-learn==0.99 처럼 말도 안되는 버전을 적으면 리스트가 뜬다
# pip install scikit-learn==1.1.3

# 데이터
datasets = load_boston()
print(datasets)
x = datasets.data
y = datasets.target
print(x)
print(x.shape)  # (506, 13) 506 행, 13 열
print(y)
print(y.shape)  # (506, ) 506 스칼라

print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']

print(datasets.DESCR)

# [실습]
# train_size 0.7 이상, 0.9 이하
# R2 0.62 이상

x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    random_state=4)

model = Sequential()
# model.add(Dense(20, input_dim = 13))    # 13개의 열을 받는다
model.add(Dense(20, input_shape = (13,)))   # 행무시 // 전체 형태의 쉐잎에서 행만을 뺀 값을 넣어준다. (10000, 100, 100, 80) 이라면
                                            # (100,100,80) 으로 인풋 쉐잎을 넣어준다.
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(9))
model.add(Dense(5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=10)
end_time = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# Epoch 500/500 - random_state = 1
# 36/36 [==============================] - 0s 525us/step - loss: 30.7250
# 5/5 [==============================] - 0s 1ms/step - loss: 20.1617
# 5/5 [==============================] - 0s 871us/step
# 16/16 [==============================] - 0s 463us/step
# 로스 :  20.161657333374023
# R2 스코어 :  0.7800254168000184


