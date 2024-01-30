# 48_2 카피해서
# (N, 4, 1) -> (N, 2, 2)
# 로 변경해서 LSTM

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping

# 1 데이터
a = np.array(range(1,101))
x_predict = np.array(range(96, 106))
size = 5        # x데이터는 4개, y데이터는 1개
print(a)
print(x_predict)
# [ 96  97  98  99 100 101 102 103 104 105]
print(a.shape)          # (100,)
print(x_predict.shape)  # (10,)

# ======================================================
def split_x(dataset, size):     # split_x 함수 정의
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)  # 잘라놓은걸 이어붙인다
    return np.array(aaa)
# ======================================================
bbb = split_x(a, size)

x = bbb[:, :-1]     # : 는 모든행, :-1 은 처음부터 마지막열 바로 앞까지
y = bbb[:, -1]      # : 는 모든행, -1 은 마지막 열 만을 하나만
print(x.shape, y.shape)         # (96, 4) (96,)
x = x.reshape(-1, 2, 2)
print(x.shape, y.shape)         # (96, 4, 1) (96,)


ccc = split_x(x_predict, 4)     # x 쉐잎의 형태를 맞추기 위해 size가 아닌 4로  
print(ccc.shape)                # (6, 4)
ccc = ccc.reshape(-1,2,2)
print(ccc.shape)                # (6, 4, 1)

# 2 모델
model = Sequential()
model.add(LSTM(units=256, return_sequences=True,
               input_shape=(2, 2))) # (timesteps, features)
model.add(LSTM(128))    # 댄스에 줄 떄 리턴스퀀스 트루 주게되면 3차원으로 들어가기 때문에 해제해야한다
model.add(Dense(32))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,))

model.summary()

# 3 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )
model.fit(x, y, epochs=100,
          callbacks=[es], batch_size=32)

#  4 평가, 예측
results  = model.evaluate(x,y)
y_pred = model.predict(ccc)
print('로스는 : ', results)
print('결과 : ', y_pred)



# 로스는 :  0.010141386650502682
# 결과 :  [[ 99.627945]
#  [100.527115]     # 96,97,98,99 4가지 숫자의 다음 숫자 100 예측
#  [101.41133 ]     # 97,98,99,100 4가지 숫자의 다음 숫자 100 예측
#  [102.2798  ]
#  [103.13176 ]
#  [103.96648 ]
#  [104.78336 ]]    # 100,101,102,103 4가지 숫자의 다음 숫자 100 예측