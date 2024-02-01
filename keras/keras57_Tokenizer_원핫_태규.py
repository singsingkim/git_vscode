# OneHotEncoder
x1=np.array(x).reshape(-1,1)
print(x1)
ohe=OneHotEncoder(sparse=False)
x1_ohe=ohe.fit_transform(x1)
print(x1_ohe.shape) #(12, 8)
x1_ohe=x1_ohe.reshape(-1,12,8)
print(x1_ohe.shape) #(1, 12, 8)

# pd.get_dummies
x2=np.array(x).reshape(-1)
x3=pd.get_dummies(x2)
# x2=pd.get_dummies(np.array(x).reshape(-1))    #위랑 똑같다.
# print(x3.shape)
x4=x3.values.reshape(-1,12,8)
# print(x4.shape)   #(1, 12, 8)

# to_categorical
x5 = to_categorical(x)
# print(x5)
# [[[0. 0. 0. 0. 1. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 1. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 1. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 1. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 1. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 1. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 1. 0. 0. 0. 0. 0. 0. 0.]
#   [0. 0. 0. 0. 0. 0. 0. 0. 1.]]]
x6=x5[:,:,1:]
print(x6.shape) #(1, 12, 9)
# to_categorical 는 0부터 시작인데 데이터는 1부터 시작 - 슬라이싱 해줘야함
# print(x1.shape)