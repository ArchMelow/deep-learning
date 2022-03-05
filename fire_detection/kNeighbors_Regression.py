import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
#회귀 문제를 해결하는 프로그램 (농어의 무게를 예측)
def draw_pyplot(x_data, y_data, xlabel, ylabel):
  plt.scatter(x_data, y_data)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.show()
def overfitting_underfitting():
  knr = KNeighborsRegressor()
  x = np.arange(5,45).reshape(-1,1)
  for n in [1,5,10]:
    knr.n_neighbors = n
    knr.fit(train_input, train_target)
    prediction = knr.predict(x)
    plt.scatter(train_input, train_target)
    plt.plot(x, prediction)
    print('----------------------- n : ',n,' -------------------------')
    plt.show()
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
draw_pyplot(perch_length, perch_weight,'length', 'weight')
train_input,test_input,train_target,test_target = train_test_split(perch_length, perch_weight, random_state = 42)
train_input = train_input.reshape(-1,1) #scikit test set must be 2-D numpy array
test_input = test_input.reshape(-1,1)
knr = KNeighborsRegressor()
knr.fit(train_input, train_target)
print('정답률 : ', knr.score(test_input, test_target)*100, '%') #R^2 (결정계수) - coefficient of determination
print('테스트 데이터 정답과 예측 데이터 사이의 평균 오차(절댓값) : ', mean_absolute_error(knr.predict(test_input), test_target))
print('훈련 세트로 평가한 R^2값 : ', knr.score(train_input, train_target)) #훈련 데이터로 평가한 R^2값
print('테스트 세트로 평가한 R^2값 : ', knr.score(test_input, test_target)) #테스트 세트로 평가한 R^2값
#테스트 세트가 훈련 세트 평가값보다 더 크게 차이나므로 이 경우 '과소적합'
#이 경우 모델을 더 복잡하게 해야 한다. 두 평가값의 차이를 줄이는 것이 관건
#K-Neighbors 알고리즘의 경우 n_neighbors=5(default)에서 3으로 줄여 주면 훈련 세트의 국지적인 패턴에 민감해져 모델이 더 복잡해진다.
knr.n_neighbors = 3
knr.fit(train_input, train_target)
print('모델 조정 후 훈련 세트로 평가한 R^2값 : ', knr.score(train_input, train_target))
print('모델 조정 후 테스트 세트로 평가한 R^2값 : ', knr.score(test_input, test_target))
overfitting_underfitting()