# 14_1 카피
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


# 현재 사이킷런 버전 1.3.0 보스턴 안됨. 그래서 삭제
# pip uninstall scikit-learn
# pip uninstall scikit-learn-intelex
# pip uninstall scikit-image

# 설치하고 싶은 pip 가 있으면 pip install scikit-learn==0.9999 처럼 말도 안되는 버전을 적으면 리스트가 뜬다
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

x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            train_size=0.7,
            # test_size=0.3,
            # shuffle=True,
            random_state=4
            )

model = Sequential()
model.add(Dense(64, input_dim = 13))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
start_time = time.time()

from keras.callbacks import EarlyStopping       # 클래스는 정의가 필요
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
            mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
            patience=10,      # 최소값 찾은 후 열 번 훈련 진행
            verbose=1,
            restore_best_weights=True   # 디폴트는 False    # 페이션스 진행 후 최소값을 최종값으로 리턴 
            )

hist = model.fit(x_train, y_train, epochs=200, 
            batch_size=1, validation_split=0.2, 
            verbose=1, callbacks=[es]     # 리스트 형태임 --> 친구가 있다
            )

end_time = time.time()

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
result = model.predict(x)

r2 = r2_score(y_test, y_predict)
print("로스 : ", loss)
print("R2 스코어 : ", r2)

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
# from matplotlib import font_manager
plt.figure(figsize=(9, 6))
plt.plot(hist.history['loss'], c='red', label='loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.plot(hist.history['val_loss'], c='blue', label='val_loss', marker='.')    # plot 을 scatter 로 바ㅏ꾸면 점으로 실제 데이터가 직선으로 찍힘
plt.legend(loc='upper right')           # 오른쪽 위 라벨표시

# font_path = "C:/Windows/Fonts/NGULIM.TTF"
# font = font_manager.FontProperties(fname=font_path).get_name()
# rc('font', family=font)
plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False

plt.title('보스턴 로스')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.show()

print("RMSE : ", rmse)
print("걸린시간 : ", round(end_time - start_time, 2),"초")

# Epoch 500/500 - random_state = 1
# 36/36 [==============================] - 0s 525us/step - loss: 30.7250
# 5/5 [==============================] - 0s 1ms/step - loss: 20.1617
# 5/5 [==============================] - 0s 871us/step
# 16/16 [==============================] - 0s 463us/step
# 로스 :  20.161657333374023
# R2 스코어 :  0.7800254168000184


