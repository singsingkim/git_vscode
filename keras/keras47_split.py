import numpy as np

a = np.array(range(1,11))
size = 5
print(a)
# [ 1  2  3  4  5  6  7  8  9 10]

def split_x(dataset, size):     # split_x 함수 정의
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)  # 잘라놓은걸 이어붙인다
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape)    # (6, 5)

'''
import numpy as np

a = np.array(range(1,11))
size = 5
print(a)
# [ 1  2  3  4  5  6  7  8  9 10]

def spx(dt, size):          # spx 함수 정의, dt 라는 공백의 변수값 생성
    aaa = []                # aaa 라는 공백의 배열을 생성           
    for i in range(len(dt) - size + 1):
        subset = dt[i : (i + size)]
        aaa.append(subset)  # 잘라놓은걸 이어붙인다
    return np.array(aaa)

bbb = spx(a, size)
print(bbb)
# [[ 1  2  3  4  5]
#  [ 2  3  4  5  6]
#  [ 3  4  5  6  7]
#  [ 4  5  6  7  8]
#  [ 5  6  7  8  9]
#  [ 6  7  8  9 10]]
print(bbb.shape)    # (6, 5)

'''