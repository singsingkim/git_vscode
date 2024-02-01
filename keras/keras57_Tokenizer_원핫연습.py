# 자연어처리    # 토큰 : 조각조각 짜른다    # 토크나이저 사요하면 원핫 필요
import numpy as np
from keras.preprocessing.text import Tokenizer

text = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'

token = Tokenizer()
token.fit_on_texts([text])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '나는': 4, '맛있는': 5, '밥을': 6, '엄청': 7, '먹었다': 8}
# 제일 많은 마구 1 / 같은 두 개중에 앞에 있는 진짜 2

print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 3), ('먹었다', 1)])

x = token.texts_to_sequences([text])
print(x)
# [[4, 2, 2, 3, 3, 5, 6, 7, 1, 1, 1, 8]]


# 1 케라스 to_categorical 에서 첫번쨰 0 빼기
from keras.utils import to_categorical  # 0번부터 시작하는 문제가 발생
x_ohe1 = to_categorical(x)
x_ohe1 = x_ohe1[:,:,1:] # 첫번째 0 빼기
print(x_ohe1)
# [[[0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1.]]]
print(x_ohe1.shape)     # (1, 12, 8)


# # 2 사이킷런 원핫인코더
from sklearn.preprocessing import OneHotEncoder
x = np.array(x).reshape(-1,1)
ohe2 = OneHotEncoder(sparse=False)
x_ohe2 = ohe2.fit_transform(x)
print(x_ohe2)
# [[0. 0. 0. 1. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1.]]
print(x_ohe2.shape)
# (12, 8)

# 3 판다스 겟더미
import pandas as pd
x = np.array(x).reshape(-1,)
x_ohe3 = pd.get_dummies(x.dtype(int))
print(x_ohe3)
#         1      2      3      4      5      6      7      8
# 0   False  False  False   True  False  False  False  False
# 1   False   True  False  False  False  False  False  False
# 2   False   True  False  False  False  False  False  False
# 3   False  False   True  False  False  False  False  False
# 4   False  False   True  False  False  False  False  False
# 5   False  False  False  False   True  False  False  False
# 6   False  False  False  False  False   True  False  False
# 7   False  False  False  False  False  False   True  False
# 8    True  False  False  False  False  False  False  False
# 9    True  False  False  False  False  False  False  False
# 10   True  False  False  False  False  False  False  False
# 11  False  False  False  False  False  False  False   True
print(x_ohe3.shape)
# (12, 8)