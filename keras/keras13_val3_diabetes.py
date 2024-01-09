from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score    # 결정계수
import time

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.7,
                                                    test_size = 0.3,
                                                    shuffle = True,
                                                    random_state = 1004)

print(x)
print(y)
print(x.shape, y.shape)     # (442, 10) (442,)
print(datasets.feature_names)   # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

# 만들기
# R2 0.62 이상

#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim = 10))
model.add(Dense(20))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(7))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
model.fit(x_train, y_train, epochs = 500, batch_size = 10, validation_split=0.3, verbose=1)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)
r2 = r2_score(y_test, y_predict)    # 결정계수

print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2),"초")

# Epoch 500/500
# 78/78 [==============================] - 0s 667us/step - loss: 2872.6047
# 5/5 [==============================] - 0s 997us/step - loss: 3197.0227
# 5/5 [==============================] - 0s 256us/step
# 14/14 [==============================] - 0s 0s/step
# 로스 :  3197.022705078125
# R2 스코어 :  0.5341697876651813
# 걸린 시간 :  21.62 초

# validation
# Epoch 500/500
# 22/22 [==============================] - 0s 1ms/step - loss: 2711.0061 - val_loss: 3274.8269
# 5/5 [==============================] - 0s 750us/step - loss: 3280.0452
# 5/5 [==============================] - 0s 500us/step
# 14/14 [==============================] - 0s 623us/step
# 로스 :  3280.045166015625
# R2 스코어 :  0.531969253894929
# 걸린 시간 :  16.37 초




