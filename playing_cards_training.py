import os
import cv2
from tensorflow import keras
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model
# from keras.preprocessing import image
import keras.utils as image
import tensorflow as tf
import numpy as np
import splitfolders
base_dir = '/home/dhanish/PycharmProjects/Images'
class_names = os.listdir(base_dir)
print(class_names)
num_classes = len(class_names)
size = 128
# splitting the data into train, test and validation using splitfolders
splitfolders.ratio("Your base directory", output="output",
    seed=1337, ratio=(.8, .1, .1), group_prefix=None, move=False)
train_dir = 'Your train directory'
val_dir = 'Your validation directory'
test_dir = 'Your test directory'
#
data_train = []
label_train = []
data_val = []
label_val = []
data_test = []
label_test = []
#
size = 128 #Crop the image to 128x128

for folder in os.listdir(train_dir):
    print(folder)
    for file in os.listdir(os.path.join(train_dir, folder)):
        print(file)
        if file.endswith("jpg"):
            label_train.append(folder)
            print(label_train)
            img = cv2.imread(os.path.join(train_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (size, size))
            data_train.append(im)
        else:
            continue



for folder in os.listdir(val_dir):
    print(folder)
    for file in os.listdir(os.path.join(val_dir, folder)):
        print(file)
        if file.endswith("jpg"):
            label_val.append(folder)
            print(label_val)
            img = cv2.imread(os.path.join(val_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (size, size))
            data_val.append(im)
        else:
            continue

for folder in os.listdir(test_dir):
    print(folder)
    for file in os.listdir(os.path.join(test_dir, folder)):
        print(file)
        if file.endswith("jpg"):
            label_test.append(folder)
            print(label_test)
            img = cv2.imread(os.path.join(test_dir, folder, file))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            im = cv2.resize(img_rgb, (size, size))
            data_test.append(im)
        else:
            continue
#
data_train_arr = np.array(data_train)
label_train_arr = np.array(label_train)
X = data_train_arr/255
y = label_train_arr
y_encoded = to_categorical(y, num_classes=num_classes)

data_val_arr = np.array(data_val)
label_val_arr = np.array(label_val)
X_val = data_val_arr/255
y_val = label_val_arr
y_val_encoded = to_categorical(y_val, num_classes=num_classes)

data_test_arr = np.array(data_test)
label_test_arr = np.array(label_test)
X_test = data_test_arr/255
y_test = label_test_arr
y_test_encoded = to_categorical(y_test, num_classes=num_classes)



model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(10, (3, 3), activation='relu', input_shape=(size, size, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(14, activation='softmax')
])



model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.summary()
history = model.fit(X,y_encoded, epochs=10, batch_size=32, validation_data=(X_val, y_val_encoded))

test_loss, test_acc = model.evaluate(X_test, y_test_encoded, verbose=1)
print("Test accuracy: ", test_acc)
# save the model
model.save('my_model.h5')