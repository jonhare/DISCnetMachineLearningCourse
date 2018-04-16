import sys
from keras.models import load_model
from scipy.misc import imread

# load a model
model = load_model('bettercnn.h5')

# load an image
image = imread(sys.argv[1]).astype(float)

# normalise it in the same manner as we did for the training data
image = image / 255.0

#reshape
image = image.reshape(1,28,28,1)

# forward propagate and print index of most likely class 
# (for MNIST this corresponds one-to-one with the digit)
print("predicted digit: "+str(model.predict_classes(image)[0]))