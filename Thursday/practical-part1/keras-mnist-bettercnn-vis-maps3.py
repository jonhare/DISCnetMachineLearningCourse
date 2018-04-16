from keras.models import load_model
from keras import backend as K
from scipy.misc import imread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# load a model
model = load_model('bettercnn.h5')

# load an image
image = imread("1.PNG").astype(float)

# normalise it in the same manner as we did for the training data
image = image / 255.0

# reshape
image = image.reshape(1,28,28,1)

# define a keras function to extract the 3rd layer response maps
get_3rd_layer_output = K.function([model.layers[0].input],
                                  [model.layers[2].output])
layer_output = get_3rd_layer_output([image])[0]

# plot the results
for i in xrange(0,15):
	plt.subplot(4,4,i+1)
	plt.imshow(layer_output[0,:,:,i], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()
plt.savefig("maps.png")