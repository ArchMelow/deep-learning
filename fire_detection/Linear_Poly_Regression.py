import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#kneighbors regression이 갖는 한계와 이를 해결하기 위한 linear regression/ polynomial regression
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
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state=42)
train_input, test_input = train_input.reshape(-1,1), test_input.reshape(-1,1)
knr = KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input, train_target)
print('50cm인 빙어(perch)의 무게 예측값(K-neighbors regression을 이용) : ', knr.predict([[50]]))
print('100cm인 빙어(perch)의 무게 예측값(K-neighbors regression을 이용) : ', knr.predict([[100]]))
# 두 경우 모두 다 1033.3333..g이 나온다 (kneighbors regression이 가장 가까운 세 개의 샘플값 평균으로 예측값을 내놓기 때문)
distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(50,1033,marker= '^') #1033은 50cm인 빙어의 무게 예측값(버림)
plt.show()
lr = LinearRegression()
lr.fit(train_input, train_target)
print('50cm인 빙어(perch)의 무게 예측값(linear regression을 이용) : ', lr.predict([[50]]))
a,b = lr.coef_, lr.intercept_ #y = ax+ b
plt.scatter(train_input, train_target)
plt.plot([15,50], [15*a+b, 50*a+b]) #y=ax+b 그림
plt.scatter(50, 1241.8, marker = '^')
plt.show()
print('훈련 세트에서의 R^2 (linear regression) : ', lr.score(train_input, train_target))
print('테스트 세트에서의 R^2 (linear regression) : ', lr.score(test_input, test_target))
#linear regression에서 일차함수 그래프를 보면 빙어의 길이가 15 이하일 때 y값이 음수이다 (물고기 무게가 음수일 리가 없다)
pr = LinearRegression() #polynomial regression
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
pr.fit(train_poly, train_target)
print('50cm인 빙어(perch)의 무게 예측값(polynomial regression을 이용) : ', pr.predict([[50**2, 50]]))
[a,b],c = pr.coef_, pr.intercept_ #y=ax^2+bx+c
point = np.arange(15,50) #15<=x<50, x= 정수
plt.scatter(train_input, train_target)
plt.plot(point, a*point**2+b*point+c) #plt.plot arg1 = x 범위, arg2 = y 범위
plt.scatter(50, 1574, marker = '^')
plt.show()
print('훈련 세트에서의 R^2 (polynomial linear regression) : ', pr.score(train_poly, train_target))
print('테스트 세트에서의 R^2 (polynomial linear regression) : ', pr.score(test_poly, test_target))