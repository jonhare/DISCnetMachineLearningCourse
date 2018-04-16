from keras.models import load_model
from scipy.misc import imread
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# load a model
model = load_model('bettercnn.h5')

weights = model.layers[0].get_weights()[0]

# plot the first layer features
for i in xrange(0,30):
	plt.subplot(5,6,i+1)
	plt.imshow(weights[:,:,0,i], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()
plt.savefig("filters.png")