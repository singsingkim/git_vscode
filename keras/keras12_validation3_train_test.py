import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

# x_train = x[:10]
# y_train = y[:10]

# x_val = x[10:13]
# y_val = y[10:13]

# x_test = x[13:16]
# y_test = y[13:16]

# train_test_split 으로만 사용해서 잘라라

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=False, train_size=0.625, random_state=1)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, shuffle=False, test_size=0.5, random_state=1)


print(x_train)
print(x_val)
print(x_test)

print(y_train)
print(y_val)
print(y_test)
# print(y_train)
# print(y_val)
# print(y_test)

#2. 모델구성
#3. 컴파일, 훈련
#4. 평가, 예측