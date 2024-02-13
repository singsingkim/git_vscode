import numpy as np
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import  HalvingGridSearchCV
# 익스페리먼트 이네이블과 쌍으로 사용해야한다
import time

# 1 데이터
x, y = load_digits(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, random_state=123, train_size=0.8, stratify=y
    )

print(x.shape)

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
# model = SVC(C=1, kernel='linear', degree=3)
# model = GridSearchCV(SVC(), 
#                      parameters,
#                      cv = kfold,
#                      verbose=1,
#                     #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
#                      n_jobs=3   # 24개의 코어중 3개 사용 / 전부사용 -1
#                      )

# model = RandomizedSearchCV(SVC(), 
print('==============하빙그리드서치 시작==========================')
model = HalvingGridSearchCV(SVC(), 
                     parameters,
                     cv = kfold,
                     verbose=1,
                    #  refit=True # 디폴트 트루 # 한바퀴 돌린후 다시 돌린다
                     n_jobs=3,   # 24개의 코어중 3개 사용 / 전부사용 -1
                     random_state=123,
                     factor=3.5,
                     min_resources=150
                     )




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



# ==============하빙그리드서치 시작==========================
# n_iterations: 3
# n_required_iterations: 4  // 4 바퀴 돌리는걸 추천
# n_possible_iterations: 3
# min_resources_: 100           // 100개만 사용
# max_resources_: 1437
# aggressive_elimination: False
# factor: 3
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100          // cv * 2 * 라벨 갯수
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 14      // 42 / 3 = 14
# n_resources: 300      // 100 * 3 = 300
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 2
# n_candidates: 5       // 안나누어 질때까지 분할 - 상위 5개 추출
# n_resources: 900
# Fitting 5 folds for each of 5 candidates, totalling 25 fits
# 최적의 매개변수 :  SVC(C=10, gamma=0.0001)
# 최적의 파라미터 :  {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
# best_score :  0.9788019863438857
# model_score :  0.9944444444444445
# accuracy_score 0.9944444444444445
# 최적 튠 ACC :  0.9944444444444445
# 걸린신간 :  0.99 초




# 팩터 4
# ==============하빙그리드서치 시작==========================
# n_iterations: 2           // 두번 돌렸어
# n_required_iterations: 3  // 3 팩터 추천이 아니라 3바퀴 돌리는걸 추천
# n_possible_iterations: 2 // 가능한게 두번이야
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 4
# ----------
# iter: 0
# n_candidates: 42
# n_resources: 100      // cv * 2 * 라벨 갯수
# Fitting 5 folds for each of 42 candidates, totalling 210 fits
# ----------
# iter: 1
# n_candidates: 11
# n_resources: 400
# Fitting 5 folds for each of 11 candidates, totalling 55 fits
# 최적의 매개변수 :  SVC(C=10, gamma=0.0001)
# 최적의 파라미터 :  {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
# best_score :  0.9520253164556962
# model_score :  0.9944444444444445
# accuracy_score 0.9944444444444445
# 최적 튠 ACC :  0.9944444444444445
# 걸린신간 :  0.86 초