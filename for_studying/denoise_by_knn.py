#출처: '으뜸 머신러닝' kNN 모델을 통한 이미지의 노이즈 제거 구현 
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize

url = 'https://github.com/dknife/ML/raw/main/data/Proj1/40/'
imgR, imgC, channel = 24, 24, 3
images = []

for i in range(40):
  file = url+ 'img{0:02d}.jpg'.format(i+1)
  img = imread(file)
  img = resize(img, (imgR, imgC, 3))
  #print(img.shape)
  images.append(img)

def plot_images(nRow, nCol, img):
  fig = plt.figure()
  fig, ax = plt.subplots(nRow, nCol, figsize = (nCol, nRow))
  for i in range(nRow):
    for j in range(nCol):
      if nRow <= 1: axis = ax[j]
      else : axis = ax[i, j]
      axis.get_xaxis().set_visible(False)
      axis.get_yaxis().set_visible(False)
      axis.imshow(img[i*nCol+j])

x = np.array(images[:30])
x_test = np.array(images[30:])
plot_images(3, 10, x)
plot_images(1, 10, x_test)

x_noise = x+ np.random.randn(len(x), imgR, imgC, channel)*0.1
x_noise = np.clip(x_noise, 0, 1)
x_test_noise = x_test+ np.random.randn(len(x_test), imgR, imgC, channel)*0.1
x_test_noise = np.clip(x_test_noise, 0, 1)
plot_images(3, 10, x_noise)
plot_images(1, 10, x_test_noise)

#input -> (0,1) , output -> (0, 255)
x_noise_flat = x_noise.reshape(-1, imgR*imgC*channel)
x_flat = np.array(x.reshape(-1, imgR*imgC*channel)*255, dtype = np.uint)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_noise_flat, x_flat)
denoised_result = knn.predict(x_noise_flat)
denoised_result = denoised_result.reshape(-1, imgR, imgC, channel)
plot_images(3, 10, denoised_result) #images' each pixels are classified by knn model. Outputs are the labels(pixels' classes) of the noised images. 

##New images are added by simply creating more noised images from the same 40 images. (for training and testing)
n_augmentation = 100 #add x100 more
x_noise_aug = x+np.random.rand(len(x), imgR, imgC, channel)*0.2
y_label = np.array(x*255, dtype = np.uint) #maybe images can't be perfectly retrieved (without noises) because dtype = np.unit forces some float vals to same int vals (classification inprecise)
y = y_label

print(x_noise.shape)
for i in range(n_augmentation):
  noisy_data = x+np.random.randn(len(x), imgR, imgC, channel) * 0.2
  x_noise_aug = np.append(x_noise_aug, noisy_data, axis = 0)
  y = np.append(y, y_label, axis = 0)

x_noise_aug = np.clip(x_noise_aug, 0, 1)
print(x_noise_aug.shape, y.shape)

plot_images(1, 10, x_noise_aug[0:300:30]) #index 0~300 by step 30 (10 elements sliced)
x_noise_aug_flattened = x_noise_aug.reshape(-1, imgR*imgC*channel)
y_flattened = y.reshape(-1, imgR*imgC*channel)
knn.fit(x_noise_aug_flattened, y_flattened)
denoised_result = knn.predict(x_noise_flat) #test the model with x_noise_flat created before (created when no image was augmented)
denoised_result = denoised_result.reshape(-1, imgR, imgC, channel)
plot_images(3, 10, x_noise)
plot_images(3, 10, denoised_result)

rndidx = np.random.randint(0, 20)
data = x[rndidx:rndidx+10] + np.random.randn(10, imgR, imgC, channel)*.4
data = np.clip(data, 0, 1)
data_flat = data.reshape(-1, imgR*imgC*channel)
denoised = knn.predict(data_flat)
denoised = denoised.reshape(-1, imgR, imgC, channel)
denoised = np.clip(denoised, 0, 255)

plot_images(1, 10, data)
plot_images(1, 10, denoised)

denoised = knn.predict(x_test_noise.reshape(-1, imgR*imgC*channel))
denoised = denoised.reshape(-1, imgR, imgC, channel)

plot_images(1, 10, x_test_noise)
plot_images(1, 10, denoised)
#overfitting problem -> because new labels were generated just by copying 30 training data labels over and over (x100 times)
#way to solve this problem is to augment new labels by actually modifying labels(rotation, shifting...)

from keras.preprocessing.image import ImageDataGenerator
image_generator = ImageDataGenerator(
    rotation_range = 360,
    zoom_range = 0.1,
    shear_range = 0.1,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True
) #keras class iterator for generating newly augmented images

y_aug = x.reshape(-1, imgR, imgC, channel)
print(y_aug.shape)
it = image_generator.flow(y_aug) #iterator for ImageDataGenerator() class object
nData = y_aug.shape[0]

x_aug = y_aug +np.random.randn(nData, imgR, imgC, channel)*0.1
n_augmentation = 500 #augmentation by x500
for _ in range(n_augmentation):
  new_y = it.next() #iterator usage
  new_x = new_y + np.random.randn(nData, imgR, imgC, channel)*0.1
  y_aug = np.append(y_aug, new_y, axis = 0)
  x_aug = np.append(x_aug, new_x, axis = 0)#append newly generated y,x row-wise

y_aug = np.array(y_aug*255, dtype = np.uint)
y_aug = y_aug.reshape(-1, imgR, imgC, channel)
x_aug = x_aug.reshape(-1, imgR, imgC, channel)
y_aug = np.clip(y_aug, 0, 255)
x_aug = np.clip(x_aug, 0, 1)
plot_images(3, 10, y_aug[30:])

#Now, train the newly generated images with KNN model
x_aug_flat = x_aug.reshape(-1, imgR*imgC*channel)
y_aug_flat = y_aug.reshape(-1, imgR*imgC*channel)
knn.fit(x_aug_flat, y_aug_flat)
denoised = knn.predict(x_test_noise.reshape(-1, imgR*imgC*channel))
denoised = denoised.reshape(-1, imgR, imgC, channel)
plot_images(1, 10, x_test_noise)
plot_images(1, 10, denoised)

images = it.next() #load 30 images generated by ImageDataGenerator()'s iterator method
testX = images + np.random.randn(nData, imgR, imgC, channel)* 0.4
testX = np.clip(testX, 0, 1)
denoised = knn.predict(testX.reshape(-1, imgR*imgC*channel))
denoised = denoised.reshape(-1, imgR, imgC, channel)

plot_images(1, 10, testX)
plot_images(1, 10, denoised)