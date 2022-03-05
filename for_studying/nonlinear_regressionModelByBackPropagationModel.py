import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = 'https://github.com/dknife/ML/raw/main/data/nonlinear.csv'
df = pd.read_csv(data)
plt.scatter(df['x'], df['y'])
plt.show()
nx, nh1, nh2, ny = 1, 6, 4, 1
U = np.random.randn(nx, nh1)*2
V = np.random.randn(nh1, nh2)*2
W = np.random.randn(nh2, ny)*2
learning_rate = 0.1
def sigmoid(v): 
  return 1/(1+np.exp(-v))
input = np.zeros(nx)
h1_out, h1_deriv = np.zeros(nh1), np.zeros(nh1)
h1_delta = np.zeros(nh1)

h2_out, h2_deriv = np.zeros(nh2), np.zeros(nh2)
h2_delta = np.zeros(nh2)

y_out, y_deriv = np.zeros(ny), np.zeros(ny)
y_delta = np.zeros(ny)

def forward(x):
  global input, h1_out, h1_deriv, h2_out, h2_deriv, y_out, y_deriv
  input = x
  h1_out = sigmoid(U.T*input)
  h1_deriv = h1_out * (1- h1_out)

  h2_out = sigmoid(V.T@h1_out)
  h2_deriv = h2_out * (1- h2_out)
  
  y_out = sigmoid(W.T@h2_out)
  y_deriv = y_out * (1- y_out)

def compute_error(target):
  return y_out - target

def backward(error):
  global y_delta, W, h2_delta, V, h1_delta, U

  y_delta = y_deriv* error
  dw = -learning_rate *np.outer(h2_out, y_delta)
  W = W+dw
  
  h2_delta = h2_deriv * W@y_delta
  dv = -learning_rate *np.outer(h1_out, h2_delta)
  V = V+dv

  h1_delta = h1_deriv * V@h2_delta
  du = -learning_rate *np.outer(input, h1_delta)
  U = U+du

def train(x, target):
  forward(x)
  e = compute_error(target)
  backward(e)
  return e**2

loss = []
X = df['x'].to_numpy()
y_label = df['y'].to_numpy()
for i in range(1000):
  e_accum = 0
  for x,y in zip(X,y_label):
    e_accum+= train(x,y)
  loss.append(e_accum)

err_log = np.array(loss).flatten()
plt.plot(err_log)
plt.show()

def predict(X):
  y_hat = []
  for x in X:
    forward(x)
    y_hat.append(y_out)
  return y_hat

domain = np.linspace(0, 1, 100).reshape(-1,1)
y_hat = predict(domain)
plt.scatter(df['x'], df['y'])
plt.scatter(domain, y_hat, color='r')