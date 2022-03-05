import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#kneighbors regression�� ���� �Ѱ�� �̸� �ذ��ϱ� ���� linear regression/ polynomial regression
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
print('50cm�� ����(perch)�� ���� ������(K-neighbors regression�� �̿�) : ', knr.predict([[50]]))
print('100cm�� ����(perch)�� ���� ������(K-neighbors regression�� �̿�) : ', knr.predict([[100]]))
# �� ��� ��� �� 1033.3333..g�� ���´� (kneighbors regression�� ���� ����� �� ���� ���ð� ������� �������� ������ ����)
distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes], marker = 'D')
plt.scatter(50,1033,marker= '^') #1033�� 50cm�� ������ ���� ������(����)
plt.show()
lr = LinearRegression()
lr.fit(train_input, train_target)
print('50cm�� ����(perch)�� ���� ������(linear regression�� �̿�) : ', lr.predict([[50]]))
a,b = lr.coef_, lr.intercept_ #y = ax+ b
plt.scatter(train_input, train_target)
plt.plot([15,50], [15*a+b, 50*a+b]) #y=ax+b �׸�
plt.scatter(50, 1241.8, marker = '^')
plt.show()
print('�Ʒ� ��Ʈ������ R^2 (linear regression) : ', lr.score(train_input, train_target))
print('�׽�Ʈ ��Ʈ������ R^2 (linear regression) : ', lr.score(test_input, test_target))
#linear regression���� �����Լ� �׷����� ���� ������ ���̰� 15 ������ �� y���� �����̴� (����� ���԰� ������ ���� ����)
pr = LinearRegression() #polynomial regression
train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))
pr.fit(train_poly, train_target)
print('50cm�� ����(perch)�� ���� ������(polynomial regression�� �̿�) : ', pr.predict([[50**2, 50]]))
[a,b],c = pr.coef_, pr.intercept_ #y=ax^2+bx+c
point = np.arange(15,50) #15<=x<50, x= ����
plt.scatter(train_input, train_target)
plt.plot(point, a*point**2+b*point+c) #plt.plot arg1 = x ����, arg2 = y ����
plt.scatter(50, 1574, marker = '^')
plt.show()
print('�Ʒ� ��Ʈ������ R^2 (polynomial linear regression) : ', pr.score(train_poly, train_target))
print('�׽�Ʈ ��Ʈ������ R^2 (polynomial linear regression) : ', pr.score(test_poly, test_target))