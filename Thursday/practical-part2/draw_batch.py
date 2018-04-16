# Plot ad hoc data instances
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
        'data/train',
        target_size=(240, 800),
        batch_size=4,
        class_mode='categorical')

# generate the first batch
(batch_images, batch_labels) = generator.next()

class_labels = [item[0] for item in sorted(generator.class_indices.items(), key=lambda x: x[1])] #get a list of classes
batch_labels = numpy.argmax(batch_labels, axis=1) #convert the one-hot labels to indices

# plot 4 images
plt.subplot(221).set_title(class_labels[batch_labels[0]])
plt.imshow(batch_images[0], aspect='equal')
plt.subplot(222).set_title(class_labels[batch_labels[1]])
plt.imshow(batch_images[1], aspect='equal')
plt.subplot(223).set_title(class_labels[batch_labels[2]])
plt.imshow(batch_images[2], aspect='equal')
plt.subplot(224).set_title(class_labels[batch_labels[3]])
plt.imshow(batch_images[3], aspect='equal')

# show the plot
plt.show()
plt.savefig("batch.png")