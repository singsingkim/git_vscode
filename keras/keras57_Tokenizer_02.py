# 자연어처리    # 토큰 : 조각조각 짜른다    # 토크나이저 사요하면 원핫 필요
import numpy as np
from keras.preprocessing.text import Tokenizer

text1 = '나는 진짜 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
text2 = '상헌이가 선생을 괴롭힌다. 상헌이는 잘생겼다. 상헌이는 마구 마구 잘생겼다.'

#### 아래 수정해봐!!!

token = Tokenizer()
token.fit_on_texts([text1, text2])  # 아래 두개와 같다
# token.fit_on_texts([text1])
# token.fit_on_texts([text2])

print(token.word_index)
# {'마구': 1, '진짜': 2, '매우': 3, '상헌이는': 4, '잘생겼다': 5, '나는': 6, '맛있는': 7, '밥을': 8, '엄청': 9, '먹었다': 10, '상헌이가': 11, '선생을': 12, '괴롭힌다': 13}

print(token.word_counts)
# OrderedDict([('나는', 1), ('진짜', 2), ('매우', 2), ('맛있는', 1), ('밥을', 1), ('엄청', 1), ('마구', 5), ('먹었다', 1), ('상헌이가', 1), ('선생을', 1), ('괴롭힌다', 1), ('상헌이는', 2), ('잘생겼다', 2)])

x = token.texts_to_sequences([text1 + text2])
print(x)
# [[6, 2, 2, 3, 3, 7, 8, 9, 1, 1, 1, 10, 11, 12, 13, 4, 5, 4, 1, 1, 5]]

# 1 케라스 to_categorical 에서 첫번쨰 0 빼기
from keras.utils import to_categorical  # 0번부터 시작하는 문제가 발생
x_ohe1 = to_categorical(x)
x_ohe1 = x_ohe1[:,:,1:] # 첫번째 0 빼기
print(x_ohe1)
# [[[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]]
print(x_ohe1.shape)     
# (1, 21, 13)

# # 2 사이킷런 원핫인코더
from sklearn.preprocessing import OneHotEncoder
x = np.array(x).reshape(-1,1)
ohe2 = OneHotEncoder(sparse=False)
x_ohe2 = ohe2.fit_transform(x)
print(x_ohe2)
# [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
print(x_ohe2.shape)
# (21, 13)
x_ohe2 = x_ohe2.reshape(-1,21,13)
print(x_ohe2.shape)
# (1, 21, 13)

# 3 판다스 겟더미
import pandas as pd
x = np.array(x).reshape(-1,)
x_ohe3 = pd.get_dummies(x)
print(x_ohe3)
#        1      2      3      4      5      6      7      8      9      10     11     12     13
# 0   False  False  False  False  False   True  False  False  False  False  False  False  False
# 1   False   True  False  False  False  False  False  False  False  False  False  False  False
# 2   False   True  False  False  False  False  False  False  False  False  False  False  False
# 3   False  False   True  False  False  False  False  False  False  False  False  False  False
# 4   False  False   True  False  False  False  False  False  False  False  False  False  False
# 5   False  False  False  False  False  False   True  False  False  False  False  False  False
# 6   False  False  False  False  False  False  False   True  False  False  False  False  False
# 7   False  False  False  False  False  False  False  False   True  False  False  False  False
# 8    True  False  False  False  False  False  False  False  False  False  False  False  False
# 9    True  False  False  False  False  False  False  False  False  False  False  False  False
# 10   True  False  False  False  False  False  False  False  False  False  False  False  False
# 11  False  False  False  False  False  False  False  False  False   True  False  False  False
# 12  False  False  False  False  False  False  False  False  False  False   True  False  False
# 13  False  False  False  False  False  False  False  False  False  False  False   True  False
# 14  False  False  False  False  False  False  False  False  False  False  False  False   True
# 15  False  False  False   True  False  False  False  False  False  False  False  False  False
# 16  False  False  False  False   True  False  False  False  False  False  False  False  False
# 17  False  False  False   True  False  False  False  False  False  False  False  False  False
# 18   True  False  False  False  False  False  False  False  False  False  False  False  False
# 19   True  False  False  False  False  False  False  False  False  False  False  False  False
# 20  False  False  False  False   True  False  False  False  False  False  False  False  False
print(x_ohe3.shape)
# (21, 13)
x_ohe3 = x_ohe3.values.reshape(-1,21,13)
print(x_ohe3.shape)
# (1, 21, 13)
