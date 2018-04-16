from keras.models import load_model
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# load a model
model = load_model('bettercnn.h5')

input_img = model.input

step=1

# we're interested in maximising outputs of the 3rd layer:
layer_output = model.layers[3].output

for i in xrange(0,15):
	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	loss = K.mean(layer_output[:, :, :, i])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])

	# we start from a gray image with some noise
	input_img_data = np.random.random((1, 28, 28, 1)) * 0.07 + 0.5
	
	# run gradient ascent for 50 steps
	for j in range(50):
		loss_value, grads_value = iterate([input_img_data])
		input_img_data += grads_value * step

	# plot the results
	plt.subplot(4,4,i+1)
	plt.imshow(input_img_data[0,:,:,0], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()
plt.savefig("maxact.png")
