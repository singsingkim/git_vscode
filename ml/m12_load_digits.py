import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.datasets import load_digits
import pandas as pd

# 1 데이터
datasets = load_digits()

x = datasets.data
y = datasets.target

print(x)
print(y)    # [0 1 2 ... 8 9 8]
print(x.shape)  # (1797, 64)    # 64니까 8 * 8
print(y.shape)  # (1797,)
print(pd.value_counts(y, sort=False))


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':['linear'], 'degree':[3, 4, 5]},  # 6 번
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},      # 12번
    {'C':[1, 10, 100, 1000], 'kernel':['sigmoid'],                      # 24번
            'gamma':[0.01, 0.001, 0.0001], 'degree':[3, 4]},
]

# 그리드로 돌렸을때 5 * 42 횟수.

# 2 모델
model = SVC(C=1, kernel='linear', degree=3)
model = GridSearchCV(SVC(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
                     )

# model = RandomizedSearchCV(SVC(), 
#                      parameters,
#                      cv = kfold,
#                      verbose=1,
#                     #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
#                      n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
#                      )




start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time()

print('최적의 매개변수 : ', model.best_estimator_)
# 최적의 매개변수 :  SVC(C=1, kernel='linear')
print('최적의 파라미터 : ', model.best_params_) # 내가 선택한것
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'} 우리가 지정한거중에 가장 좋은거
print('best_score : ', model.best_score_)   # 핏한거의 최고의 스코어
# best_score :  0.975
print('model_score : ', model.score(x_test, y_test))    # 
# model_score :  0.9666666666666667


y_predict = model.predict(x_test)
print('accuracy_score', accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
            # SVC(C-1, kernel='linear').predicict(x_test)
print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print('걸린신간 : ', round(end_time - start_time, 2), '초')

# import pandas as pd
# print(pd.DataFrame(model.cv_results_).T)

# 그리드서치
# 최적 튠 ACC :  0.9916666666666667
# 걸린신간 :  2.51 초


# 랜덤 서치
# 최적 튠 ACC :  0.9916666666666667
# 걸린신간 :  1.05 초