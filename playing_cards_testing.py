from keras.models import load_model
import keras.utils as image
import numpy as np
size = 128


# load the saved model
loaded_model = load_model('my_model.h5')

img = image.load_img('Your playing card test image', target_size=(size, size))
# to sacle the image
img = image.img_to_array(img)
img = img/255

# convert the PIL Image object to a NumPy array
img_array = image.img_to_array(img)

# add an extra dimension to represent the batch size
img_array = np.expand_dims(img_array, axis=0)

# make a prediction using the model
out = loaded_model.predict(img_array, verbose=1)

# get the predicted class
pred_out = np.argmax(out, axis=1)
print(pred_out)