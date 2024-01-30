# 열 14개
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time
start_time = time.time()


#1. 데이터
path = "C:/_data/kaggle/jena//"
datasets = pd.read_csv(path + "jena_climate_2009_2016.csv", index_col=0 )
print(datasets.shape)  # (420551, 14)
print(datasets.columns)
# Index(['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)',
#        'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
#        'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
#        'wd (deg)'],
#       dtype='object')
print(datasets.dtypes)
# p (mbar)           float64
# T (degC)           float64
# Tpot (K)           float64
# Tdew (degC)        float64
# rh (%)             float64
# VPmax (mbar)       float64
# VPact (mbar)       float64
# VPdef (mbar)       float64
# sh (g/kg)          float64
# H2OC (mmol/mol)    float64
# rho (g/m**3)       float64
# wv (m/s)           float64
# max. wv (m/s)      float64
# wd (deg)           float64
# dtype: object



# split_x 함수
# ======================================================
def split_x(datasets, size, y):     # split_x 함수 정의
    aaa = []
    bbb = []
    
    num = len(datasets) - size
    for i in range(num):
        aaa.append(datasets[i : i + size])
        y_row = datasets.iloc[i+size]
        bbb.append(y_row[y])

    return np.array(aaa), np.array(bbb)
# ======================================================
x, y = split_x(datasets, 3, 'T (degC)')
print(x.shape, y.shape)        # (420548, 3, 14) (420548,)


# 트레인테스트스플릿
x_train, x_test, y_train, y_test = train_test_split(
            x, y, train_size=0.8, shuffle=False, random_state=123
)


#2. 모델 구성 
model = Sequential()
model.add(LSTM(units=256, return_sequences=True,
               input_shape=(3, 14))) # (timesteps, features)
model.add(LSTM(128))    # 댄스에 줄 떄 리턴스퀀스 트루 주게되면 3차원으로 들어가기 때문에 해제해야한다
model.add(Dense(32))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8, activation='relu'))
model.add(Dense(1,))

model.summary()


#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['acc'])
es = EarlyStopping(monitor='val_loss',
                mode='min',
                patience=500,
                verbose=1,
                restore_best_weights=True
                )

model.fit(x_train, y_train, epochs=5000, 
                batch_size = 4096,
                validation_split=0.2,
                callbacks=[es],
                verbose=1
                )

end_time = time.time()   #끝나는 시간

#4.평가, 예측
results = model.evaluate(x_train, y_train)
y_predict = model.predict(x_test)
print("걸린시간 : ", round(end_time - start_time, 2),"초")
print('로스 : ', results[0])
print('acc : ', results[1])


