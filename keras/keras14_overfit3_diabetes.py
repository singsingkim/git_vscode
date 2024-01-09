from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score, mean_squared_error    # 결정계수
import numpy as np
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
model.add(Dense(64, input_dim = 10))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer = 'adam')
start_time = time.time()
hist = model.fit(x_train, y_train, epochs = 10, batch_size = 10, validation_split=0.3, verbose=1)
end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)
r2 = r2_score(y_test, y_predict)    # 결정계수

print("로스 : ", loss)
print("R2 스코어 : ", r2)
print("걸린 시간 : ", round(end_time - start_time,2),"초")

def RMSE(aaa, bbb):
    return np.sqrt(mean_squared_error(aaa, bbb))
rmse = RMSE(y_test, y_predict)

print("RMSE : ", rmse)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

print("============ hist =============")
print(hist)
print("===============================")
# print(datasets) #데이터.데이터 -> 데이터.타겟
print(hist.history)     # 오늘과제 : 리스트, 딕셔너리=키(loss) : 똔똔 밸류 한 쌍괄호{}, 튜플
                                    # 두 개 이상은 리스트
                                    # 딕셔너리
print("============ loss =============")
print(hist.history['loss'])
print("============ val_loss =========")
print(hist.history['val_loss'])
print("===============================")

# ★ 시각화 ★
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)

plt.title('당뇨병 로스')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

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




