import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

x_train = x[:10]
y_train = y[:10]

x_val = x[10:13]
y_val = y[10:13]

x_test = x[13:16]
y_test = y[13:16]

print(x_train)
print(x_val)
print(x_test)
print(y_train)
print(y_val)
print(y_test)

# print(x_val)
# print(x_test)

#2. 모델구성
#3. 컴파일, 훈련
#4. 평가, 예측