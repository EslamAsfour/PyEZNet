import numpy as np
import DataLoader


print('Loadind data......')
num_classes = 10
train_images = DataLoader.train_images() #[60000, 28, 28]
train_images= np.array(train_images,dtype=np.float64)
train_labels = DataLoader.train_labels()
test_images = DataLoader.test_images()
test_images= np.array(test_images,dtype=np.float64)
test_labels = DataLoader.test_labels()

print('Preparing data......')
train_images -= int(np.mean(train_images))
train_images /= int(np.std(train_images))
test_images -= int(np.mean(test_images))
test_images /= int(np.std(test_images))
training_data = train_images.reshape(60000, 1, 28, 28)
training_labels = np.eye(num_classes)[train_labels]
testing_data = test_images.reshape(10000, 1, 28, 28)
testing_labels = np.eye(num_classes)[test_labels]