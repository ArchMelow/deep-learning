from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
data = load_iris()
iris_input = data['data']
iris_label = data['target']
flower_names = ['versicolor', 'setosa', 'virginica']
train_input, test_input, train_target, test_target = train_test_split(iris_input, iris_label, test_size = 0.3)
print(train_input.shape, train_target.shape, test_input.shape, test_target.shape, train_input[0].shape)
print(test_input)
print(train_input[0])
model = keras.models.Sequential([
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(10, activation = 'relu'),
    keras.layers.Dense(3, activation= 'softmax'),
])
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_input, train_target, batch_size = 5, epochs = 30)
model.evaluate(test_input, test_target, verbose = 2)
plt.plot(history.history['loss'], color = 'b')
plt.show()
plt.plot(history.history['accuracy'], color = 'r') 
random_idx = np.random.randint(0, 45)
pred = model.predict(test_input[random_idx][np.newaxis,:])
#print(np.argmax(pred))
print('Flower Predicted : {}'.format(flower_names[np.argmax(pred)]))
print('Answer : {}'.format(flower_names[test_target[random_idx]]))