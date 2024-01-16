# https://dacon.io/competitions/open/235610/data

import numpy as np      # 수치화 연산
import pandas as pd     # 각종 연산 ( 판다스 안의 파일들은 넘파이 형식)
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import time                
#1. 데이터
path = "c:/_data/dacon/dechul//"
train_csv = pd.read_csv(path + "train.csv", index_col = 0)  # 인덱스를 컬럼으로 판단하는걸 방지
test_csv = pd.read_csv(path + "test.csv", index_col = 0)
subm_csv = pd.read_csv(path + "sample_submission.csv")   # 여기 있는 id 는 인덱스 취급하지 않는다.

print(train_csv.shape)          # (96294, 14)
print(test_csv.shape)           # (64197, 13) // 아래 서브미션과의 열의 합이 11 인것은 id 열 이 중복되어서이다
print(subm_csv.shape)           # (64197, 2)

print(train_csv.columns)         
        # Index(['대출금액', '대출기간', '근로기간', '주택소유상태', '연간소득', '부채_대비_소득_비율', '총계좌수', '대출목적',
        #        '최근_2년간_연체_횟수', '총상환원금', '총상환이자', '총연체금액', '연체계좌수', '대출등급'],
        #       dtype='object')


######### 결측치 처리 1. 제거 #########
# train_csv = train_csv.dropna()      # 결측치가 한 행에 하나라도 있으면 그 행을 삭제한다
######### 결측치 처리 2. 0으로 #########
# train_csv = train_csv.fillna(0)   # 결측치 행에 0을 집어 넣는다

# print(train_csv.isnull().sum())
print(train_csv.isna().sum())       # 위 와 같다. isnull() = isna()
        # 대출금액            0
        # 대출기간            0
        # 근로기간            
        
        # 주택소유상태          0
        # 연간소득            0
        # 부채_대비_소득_비율     0
        # 총계좌수            0
        # 대출목적            0
        # 최근_2년간_연체_횟수    0
        # 총상환원금           0
        # 총상환이자           0
        # 총연체금액           0
        # 연체계좌수           0
        # 대출등급            0


print(train_csv.info())
        # Data columns (total 14 columns):
        #  #   Column           Non-Null Count    Dtype
        # ---  ------           --------------    -----
        #  0   대출금액           96294 non-null  int64
        #  1   대출기간           96294 non-null  object
        #  2   근로기간           96294 non-null  object
        #  3   주택소유상태       96294 non-null  object
        #  4   연간소득           96294 non-null  int64
        #  5   부채대비소득비율   96294 non-null  float64
        #  6   총계좌수           96294 non-null  int64
        #  7   대출목적           96294 non-null  object
        #  8   최근2년간연체횟수  96294 non-null  int64
        #  9   총상환원금         96294 non-null  int64
        #  10  총상환이자         96294 non-null  float64
        #  11  총연체금액         96294 non-null  float64
        #  12  연체계좌수         96294 non-null  float64
        #  13  대출등급           96294 non-null  object
print(train_csv.shape)  # (96294, 14)
print(train_csv)
        #              대출금액    대출기간   근로기간    주택소유상태  연간소득  부채대비소득비율  총계좌수   대출목적  최근2년간연체횟수    총상환원금     총상환이자  총연체금액  연체계좌수 대출등급
        # ID
        # TRAIN_00000  12480000   36 months    6 years      RENT      72000000        18.90         15      부채 통합             0                 0            0.0        0.0         0.0    C     
        # TRAIN_00001  14400000   60 months  10+ years  MORTGAGE     130800000        22.33         21      주택 개선             0             373572      234060.0        0.0         0.0    B     
        # TRAIN_00002  12000000   36 months    5 years  MORTGAGE      96000000         8.60         14      부채 통합             0             928644      151944.0        0.0         0.0    A     
        # TRAIN_00003  14400000   36 months    8 years  MORTGAGE     132000000        15.09         15      부채 통합             0             325824      153108.0        0.0         0.0    C     
        # TRAIN_00004  18000000   60 months    Unknown      RENT      71736000        25.39         19      주요 구매             0             228540      148956.0        0.0         0.0    B     
        # ...               ...         ...        ...       ...        ...          ...   ...    ...           ...      ...       ...    ...    ...  ...
        # TRAIN_96289  14400000   36 months  10+ years  MORTGAGE     210000000         9.33         33      신용 카드             0             974580      492168.0        0.0         0.0    C     
        # TRAIN_96290  28800000   60 months  10+ years  MORTGAGE     132000000         5.16         25      주택 개선             0             583728      855084.0        0.0         0.0    E     
        # TRAIN_96291  14400000   36 months     1 year  MORTGAGE      84000000        11.24         22      신용 카드             0             1489128     241236.0        0.0         0.0    A     
        # TRAIN_96292  15600000   36 months    5 years  MORTGAGE      66330000        17.30         21      부채 통합             2             1378368     818076.0        0.0         0.0    D     
        # TRAIN_96293   8640000   36 months  10+ years      RENT      50400000        11.80         14      신용 카드             0             596148      274956.0        0.0         0.0    C     


train_csv['대출기간'] =train_csv['대출기간'].map({' 36 months':36, ' 60 months':60}).astype(int)
test_csv['대출기간'] = test_csv['대출기간'].map({' 36 months':36, ' 60 months':60}).astype(int)

# train_csv['근로기간'] =train_csv['근로기간'].map({' 36 months':36, ' 60 months':60}).astype(int)
# test_csv['근로기간'] = test_csv['근로기간'].map({' 36 months':36, ' 60 months':60}).astype(int)

le = LabelEncoder()
# train_csv['근로기간'] = le.fit_transform(train_csv(['근로기간' != 'Unknown']))
print(train_csv)
train_csv['근로기간'] = le.fit_transform(train_csv['근로기간'])
print(train_csv)
# TRAIN_96292  15600000    36     8  MORTGAGE   66330000        17.30    21  부채 통합             2  1378368  818076.0    0.0    0.0    D
# TRAIN_96293   8640000    36     2      RENT   50400000        11.80    14  신용 카드             0   596148  274956.0    0.0    0.0    C

train_csv['주택소유상태'] = le.fit_transform(train_csv['주택소유상태'])

print(np.unique(train_csv['근로기간'], return_counts=True))
# array([    0,     1,     2,     3,     4,     5,     6,     7,     8,    9,    10,    11,    12,    13,    14,    15]), 
# array([ 6249,    56, 31585,   896,  8450,    89,  7581,  5588,  5665, 3874,  3814,  4888,  3744,  7774,   370,  5671], dtype=int64))
print(np.unique(train_csv['주택소유상태'], return_counts=True))
# array([0,     1,     2,     3]),
# array([1, 47934, 10654, 37705], dtype=int64))

print(pd.value_counts(train_csv['근로기간']))
        # 근로기간
        # 2     31585
        # 4      8450
        # 13     7774
        # 6      7581
        # 0      6249
        # 15     5671
        # 8      5665
        # 7      5588
        # 11     4888
        # 9      3874
        # 10     3814
        # 12     3744
        # 3       896
        # 14      370
        # 5        89
        # 1        56
print(pd.value_counts(train_csv['주택소유상태']))
        # 주택소유상태
        # 1    47934
        # 3    37705
        # 2    10654
        # 0        1

print(train_csv)

######### x 와 y 를 분리 #########
x = train_csv.drop(['대출등급'], axis = 1)
y = train_csv['대출등급']

print(np.unique(y, return_counts=True))
        # (array(['A', 'B', 'C', 'D', 'E', 'F', 'G'], dtype=object), array([16772, 28817, 27623, 13354,  7354,  1954,   420], dtype=int64))
print(pd.value_counts(y))
        # 대출등급
        # B    28817
        # C    27623
        # A    16772
        # D    13354
        # E     7354
        # F     1954
        # G      420
print(train_csv.shape)




# ========== 원 핫 인코딩 전처리 ==============
# 1) 케라스
# from keras.utils import to_categorical
# y_ohe = to_categorical(y)   # [1. 0. 0. ] 으로 표현
# print(y_ohe)
# print(y_ohe.shape)  # 

# 슬라이싱해서 0번째를 자름 / 7 을 0으로 바꿀수 있다 - 라벨값이 평등 / 0부터 라벨링이 줄어듬
# 최대값에서 플러스 일 만큼 만들어줌

# 2) 판다스
y_ohe2 = pd.get_dummies(y)  # [True  False  False] 으로 표현
print(y_ohe2)
print(y_ohe2.shape) # 


'''
# 3) 사이킷런
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()   # (sparse=False)
y = y.reshape(-1, 1)    # (행, 열) 형태로 재정의 // -1 은 열의 정수값에 따라 알아서 행을 맞추어 재정의하라 
y_ohe3 = ohe.fit_transform(y).toarray() # // 투어레이 사용하면 위에 스파라스 안씀. 스파라스 사용하면 투어레이 안씀
print(y_ohe3)
print(y_ohe3.shape) # (178, 3)
'''


x_train, x_test, y_train, y_test = train_test_split(x, y_ohe2, 
            shuffle=True, train_size= 0.7,
            random_state= 78567,
            stratify=y_ohe2)        # 스트레티파이 y_ohe2

print(np.unique(y_test, return_counts=True))
# 
print(x_train.shape, x_test.shape)  # (67405, 13) (28889, 13)
print(y_train.shape, y_test.shape)  # (67405, 7) (28889, 7)


#2. 모델구성
model = Sequential()
model.add(Dense(64, input_shape = (13, )))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(7, activation = 'softmax'))    

print(y_ohe2)
print(y_ohe2)


#3. 컴파일, 훈련
model.compile(loss = 'categorical_crossentropy', 
              optimizer = 'adam', # 이진분류는 아웃풋레이어에 액티베이션은 시그모이드 = 0 ~ 1 확정짓기위해. 히든레이어에 사용해도 가능
              metrics=['acc'])  # # accuracy = acc # 매트릭스 acc 정확도 체크. 가중치에 들어가진 않음 # 애큐러시는 시그모이드를 통해 받은 값을 0.5 를 기준으로 위 아래를 0 또는 1 로 인식한다. 이걸로 이큐러시 몇퍼센트라고 결과를 낸다.

from keras.callbacks import EarlyStopping       # 클래스는 정의가 필요
es = EarlyStopping(monitor = 'val_loss',    # 상당히 중요한 함수
                     mode = 'min',        # max 를 사용하는 경우도 있다 min, max, auto
                     patience=200,      # 최소값 찾은 후 설정값 만큼 훈련 진행  , 발로스 최소값 갱신 한도
                     verbose=1,
                     restore_best_weights=True   # 디폴트는 False # 페이션스 진행 후 최소값을 최종값으로 리턴 
                     )

hist = model.fit(x_train, y_train, epochs = 1,
                 batch_size = 200, validation_split=0.2,
                 verbose=1, callbacks=[es], )

#4. 평가, 예측
results = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
print("로스 : ", results[0])
print("ACC : ", results[1])
print(y_predict)


print(y_test)
print(y_predict.shape, y_test.shape)    # 

y_test = np.argmax(y_test, axis=1)
y_predict = np.argmax(y_predict, axis=1)


# ============= 모델을 최종적으로 완성 후 테스트값 받은 파일을 모델 돌려서 예측값을 서브밋에 뽑아낸 것===
y_submit = model.predict(test_csv)  # 테스트 파일을 모델에 넣어서 예측값을 뽑아내서. 와이서브밋에 저장한거를 아래 서브미션 씨에스브이에서 저장
# ================================================


print(y_test)
print(y_predict)

y_submit=np.argmax(y_submit, axis=1)
# y_submit=np.argmax(y_submit, axis=1)+3

######## submission.csv 만들기(컬럼에 값 넣어주기) ########
subm_csv['대출등급'] = y_submit

print(subm_csv)


# ============= 모델을 위에서 뽑아낸 것을 csv 파일로 생성===============
subm_csv.to_csv(path + "dechul_submission_0115_1.csv", index = False)

from sklearn.metrics import accuracy_score
def ACC(a,b):
    return accuracy_score(a,b)
acc = ACC(y_test, y_predict)

print("로스 : ", results[0])
print("ACC : ", results[1])
print("acc : ", acc)



# 

